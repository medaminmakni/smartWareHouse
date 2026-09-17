import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from src.vision import VisionPipeline
from src.agent import WarehouseAgent
from src.database import register_new_user, login_user
from dotenv import load_dotenv
import datetime

load_dotenv()

# --- Pydantic Models for Validation ---
class ProductCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    stock: int = Field(..., ge=0)
    price: float = Field(..., ge=0)
    
    @field_validator('name')
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Product name cannot be empty')
        return v.strip()

class ProductUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    stock: int = Field(..., ge=0)
    price: float = Field(..., ge=0)

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1)
    user_id: Optional[int] = None
    session_id: Optional[int] = None
    detected_plate: Optional[str] = None
    confirm_order_id: Optional[int] = None  # NEW: For confirmation responses

class PasswordUpdate(BaseModel):
    user_id: int
    new_password: str = Field(..., min_length=6)

class UserRegistration(BaseModel):
    fullName: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., min_length=5)
    company: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=6)
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v or '.' not in v:
            raise ValueError('Invalid email format')
        return v.strip().lower()

class UserLogin(BaseModel):
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=1)

app = FastAPI(title="SmartWarehouse AI API")

# CORS setup for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_PATH = "smartALPR_best.pt"

if not API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY not found in .env")

vision = VisionPipeline(MODEL_PATH, API_KEY)
agent = WarehouseAgent(API_KEY)

@app.post("/process-entrance")
async def process_entrance(file: UploadFile = File(...)):
    """
    End-to-end flow: Image -> OCR -> RAG Reasoning -> Decision
    """
    try:
        # Read image
        image_bytes = await file.read()
        
        # 1. Vision Layer: OCR to get Plate
        plate_number = vision.extract_plate_number(image_bytes)
        
        if not plate_number:
            return {
                "status": "error",
                "message": "No license plate detected in the image.",
                "decision": "HOLD",
                "analysis": "Vehicle arrived but plate recognition failed. Manual check required."
            }

        print(f"✅ Plate Detected: {plate_number}")

        # 2. Reasoning Layer: RAG + Agent Decision
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        analysis_raw = agent.reason({"plate": plate_number, "time": current_time})
        
        import json, re
        match = re.search(r'\{.*\}', analysis_raw, re.DOTALL)
        if match:
            decision = json.loads(match.group())
        else:
            decision = {
                "analysis": analysis_raw,
                "gate": "N/A",
                "priority": "MEDIUM",
                "action": "HOLD"
            }

        # 3. Action Layer: Update SQL Database based on detection
        from src.database import get_complete_arrival_info
        facts = get_complete_arrival_info(plate_number)

        return {
            "status": "success",
            "plate": plate_number,
            "decision": decision,
            "analysis": decision.get("analysis", ""),
            "timestamp": current_time,
            "factual_data": facts
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Database initialized on startup
from src.database import init_db
init_db()

@app.post("/chatbot-order")
async def chatbot_order(data: dict):
    """
    ✅ FIXED CHATBOT ENDPOINT WITH CONFIRMATION FLOW
    
    Critical Fixes Applied:
    - #1: Order creation requires explicit user confirmation
    - #2: Stock validation before order creation
    - #4: Exact client matching or candidate list
    - #5: Product disambiguation
    - #3: Plate source tracking (vision vs chat)
    
    Flow:
    1. User provides intent="order" with all fields
       → System returns confirmation_preview (NO order created)
    2. User replies with "yes" or confirmation
       → System executes create_new_order() and marks user_confirmed=TRUE
    """
    try:
        user_message = data.get("message", "").strip()
        user_id = data.get("user_id")
        session_id = data.get("session_id")
        detected_plate = data.get("detected_plate")
        confirm_order_id = data.get("confirm_order_id")  # For confirmation responses
        vision_reasoning = data.get("vision_reasoning")
        vision_decision = data.get("vision_decision")  # e.g., {"gate": "A-01", "action": "UNLOADING"}
        
        if not user_message:
            return {"status": "error", "message": "Message cannot be empty"}
        
        from src.database import (
            list_clients, list_products, create_new_order, create_new_client,
            save_chat_message, get_chat_messages, create_chat_session,
            get_complete_arrival_info, confirm_order, verify_plate,
            get_client_by_user, get_awaiting_orders_for_plate, confirm_pickup,
            complete_order, get_order_by_plate, get_active_orders_for_user
        )
        
        # Get session first
        if not session_id:
            title = user_message[:30] + "..." if len(user_message) > 30 else user_message
            session_id = create_chat_session(user_id or 1, title)
        
        # ====================================================================
        # SPECIAL: Handle Order Confirmation (user replies "yes" to preview)
        # ====================================================================
        if confirm_order_id and user_message.lower() in ['yes', 'confirm', 'oui', 'ok']:
            result = confirm_order(confirm_order_id, user_id)
            
            if result['success']:
                save_chat_message(session_id, "user", user_message)
                save_chat_message(session_id, "ai", f"✅ Order #{confirm_order_id} confirmed! Processing now.")
                
                return {
                    "status": "success",
                    "type": "order_confirmed",
                    "message": f"✅ Order #{confirm_order_id} confirmed and activated!",
                    "order_id": confirm_order_id,
                    "session_id": session_id
                }
            else:
                return {
                    "status": "error",
                    "type": "confirmation_failed",
                    "message": f"Could not confirm order: {result.get('error')}",
                    "session_id": session_id
                }
        
        # ====================================================================
        # MAIN: Gather context for LLM (let it be the brain)
        # ====================================================================
        available_clients = list_clients()
        available_products = list_products()
        
        # Get user's primary client (if any)
        user_client = get_client_by_user(user_id) if user_id else None
        user_client_name = user_client['nom'] if user_client else None
        user_client_id = user_client['idClient'] if user_client else None
        
        # Get user's active orders for context (pass to LLM)
        user_active_orders = []
        active_orders_text = ""
        if user_id:
            active_result = get_active_orders_for_user(user_id)
            if active_result.get("success"):
                user_active_orders = active_result.get("orders", [])
                if user_active_orders:
                    active_orders_text = "COMMANDES ACTIVES DE L'UTILISATEUR:\n"
                    for o in user_active_orders:
                        active_orders_text += f"- Commande #{o['id']}: {o['product_name']} x{o['quantity']} | Plaque: {o.get('plate') or 'N/A'}\n"
        
        # Check for existing orders for detected plate
        order_info = None
        current_order_status = None
        if detected_plate:
            order_lookup = get_order_by_plate(detected_plate)
            if order_lookup and order_lookup.get("found"):
                order_info = order_lookup["order"]
                current_order_status = order_info.get("statut")
        
        # Get conversation history for context
        chat_history = get_chat_messages(session_id) if session_id else []
        history_text = ""
        if chat_history:
            history_lines = []
            for msg in chat_history[-10:]:  # Last 10 messages for context
                role_label = "Client" if msg.get('role') == 'user' else "Assistant"
                history_lines.append(f"{role_label}: {msg.get('content', '')}")
            history_text = "\n".join(history_lines)
        
        # Build product list with details
        products_info = []
        for p in available_products:
            products_info.append(f"{p['name']} (Stock: {p.get('stock', 'N/A')}, Prix: {p.get('price', 'N/A')} TND)")
        
        # ====================================================================
        # RAG: Retrieve relevant rules from knowledge base
        # ====================================================================
        from src.rag_engine import get_engine
        from src.database import (
            get_pending_orders_today, get_sales_report_today, get_clients_visited_today
        )
        # get_engine() returns a module-level singleton. Constructing
        # WarehouseRAGEngine() here reloaded the sentence-transformer model on
        # every chat message.
        rag_engine = get_engine()

        # Two queries rather than one: the user's message retrieves topical
        # content, while a fixed query keeps the chatbot's own conduct rules in
        # context even when the message does not resemble them lexically.
        rag_hits = rag_engine.query_multi(
            [
                f"{user_message}",
                "règles du chatbot ton langue gestion des commandes "
                "identification du client comportement général",
            ],
            n_results=4,
        )
        rag_context = "\n---\n".join(doc for doc, _, _ in rag_hits[:6])
        
        # ====================================================================
        # ADMIN QUERIES: Build context for admin business flows
        # ====================================================================
        pending_orders_context = ""
        sales_report_context = ""
        clients_visited_context = ""
        
        # Pending orders today (for "commandes en attente" query)
        pending_result = get_pending_orders_today()
        if pending_result.get("success"):
            orders = pending_result.get("orders", [])
            if orders:
                pending_orders_context = "COMMANDES EN ATTENTE AUJOURD'HUI:\n"
                for o in orders:
                    pending_orders_context += f"- #{o['id']}: {o['client_name']} | {o['product_name']} x{o['quantity']} | Plaque: {o.get('plate') or 'N/A'}\n"
        
        # Sales report today (for "bilan du jour" query)
        sales_result = get_sales_report_today()
        if sales_result.get("success"):
            products = sales_result.get("products", [])
            revenue = sales_result.get("total_revenue", 0)
            if products:
                sales_report_context = "📊 BILAN VENTES AUJOURD'HUI:\n"
                for p in products:
                    sales_report_context += f"- {p['product_name']}: {p['total_qty']} unités | Revenu: {p.get('total_revenue', 0)} TND\n"
                sales_report_context += f"REVENU TOTAL: {revenue} TND\n"
        
        # Clients visited today (for "clients passés aujourd'hui" query)
        clients_result = get_clients_visited_today()
        if clients_result.get("success"):
            clients = clients_result.get("clients", [])
            if clients:
                clients_visited_context = f"CLIENTS PASSÉS AUJOURD'HUI: {', '.join(clients)}\n"
        
        context_summary = f"""Tu es l'assistant SmartWarehouse (pickup-only).
Aujourd'hui: {datetime.datetime.now().strftime('%Y-%m-%d')}

CONTEXTE UTILISATEUR:
- Client connecté: {user_client_name or 'Non connecté'}
- ID client: {user_client_id or 'N/A'}
{('- Véhicule détecté au portail: ' + detected_plate) if detected_plate else ''}

{active_orders_text if active_orders_text else ''}

IMPORTANT - RÈGLES DE STATUT (FLOW PICKUP UNIQUEMENT):
Statut workflow: awaiting_pickup → picked_up
- Si l'utilisateur demande "suivi de commande", liste ses commandes actives
- Si l'utilisateur demande du support, fournis les coordonnées officielles
- Pour confirmer un pickup, tu peux proposer les DB via des intentions spéciales

PRODUITS DISPONIBLES:
{chr(10).join([f'{p["name"]} (Stock: {p.get("stock", "N/A")}, Prix: {p.get("price", "N/A")} TND)' for p in available_products]) if available_products else 'Aucun produit'}

CONTEXTE ADMIN (REQUÊTES MÉTIER):
{pending_orders_context if pending_orders_context else "Pas de commandes en attente aujourd'hui"}
{sales_report_context if sales_report_context else "Pas de ventes aujourd'hui"}
{clients_visited_context if clients_visited_context else "Pas de clients visiteurs aujourd'hui"}

HISTORIQUE DE CONVERSATION:
{history_text if history_text else 'Nouvelle conversation'}

RÈGLES ET POLITIQUES (du système RAG):
{rag_context if rag_context else 'Pas de règles spécifiques'}

CONTEXTE TECHNIQUE:
- Le client connecté est "{user_client_name}" - utilise ce nom pour les commandes
- Pour créer une commande: produit + quantité + plaque (TOUS OBLIGATOIRES)
- Sans plaque = pas de commande
- Formats plaque: "159 تونس 8240" ou "3341323 نت"
- Retourne intent='order_preview' UNIQUEMENT avec les 3 infos

FORMAT DE RÉPONSE (JSON):
{{
    "response": "Ta réponse au client",
    "intent": "chat|order_preview",
    "details": {{ "client": "{user_client_name}", "product": "nom exact", "quantity": nombre, "plate": "plaque ou null" }}
}}

Message de l'utilisateur: "{user_message}"
"""
        
        # ====================================================================
        # CALL GEMINI LLM
        # ====================================================================
        import google.generativeai as genai
        import json
        import re
        import time
        
        # Helper function to validate and normalize Tunisian plate format
        def normalize_tunisian_plate(plate_str):
            if not plate_str:
                return None, False
            
            plate_str = plate_str.strip()
            
            # Normalize common Latin variants to Arabic
            normalized = plate_str.lower()
            normalized = re.sub(r'\b(tounes|tunisie|tunis|tun)\b', 'تونس', normalized, flags=re.IGNORECASE)
            normalized = re.sub(r'\b(rs|nt)\b', 'نت', normalized, flags=re.IGNORECASE)
            
            # Restore digits (they might have been affected)
            # Extract numbers and reconstruct
            parts = re.findall(r'[\d]+|تونس|نت', normalized)
            if parts:
                normalized = ' '.join(parts)
            
            # Format 1: [Code] تونس [4 digits] - e.g., "159 تونس 8240"
            pattern1 = r'^([\d]{1,3})\s*تونس\s*([\d]{4})$'
            # Format 2: [7 digits] نت - e.g., "3341323 نت"
            pattern2 = r'^([\d]{7})\s*نت$'
            
            match1 = re.match(pattern1, normalized)
            match2 = re.match(pattern2, normalized)
            
            if match1:
                return f"{match1.group(1)} تونس {match1.group(2)}", True
            elif match2:
                return f"{match2.group(1)} نت", True
            
            return plate_str, False
        
        max_retries = 3
        res_text = None
        
        for attempt in range(max_retries):
            try:
                gemini_chat_model = genai.GenerativeModel('models/gemini-2.5-flash')
                response = gemini_chat_model.generate_content(context_summary)
                res_text = response.text
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return {
                        "status": "error",
                        "message": "Le service est temporairement surchargé. Veuillez réessayer dans quelques secondes.",
                        "session_id": session_id
                    }
                raise
        
        if not res_text:
            return {
                "status": "error",
                "message": "Impossible de contacter le service. Veuillez réessayer.",
                "session_id": session_id
            }
        
        # Parse LLM response
        match = re.search(r'\{.*\}', res_text, re.DOTALL)
        if not match:
            return {
                "status": "error",
                "message": "Could not parse AI response",
                "session_id": session_id
            }
        
        decision = json.loads(match.group())
        intent = decision.get("intent")
        
        # Save user message
        save_chat_message(session_id, "user", user_message)
        save_chat_message(session_id, "ai", decision['response'])
        
        # ====================================================================
        # CRITICAL FIX #1: Handle intent='order_preview' (confirmation required)
        # ====================================================================
        if intent == "order_preview" and decision.get("details"):
            det = decision["details"]
            
            # Auto-fill client from logged-in user if not specified
            client_name = det.get('client') or user_client_name
            plate = det.get('plate') or detected_plate
            
            # Validate product, quantity AND plate are present
            if not all([det.get('product'), det.get('quantity'), plate]):
                return {
                    "status": "chat",
                    "message": decision['response'],
                    "session_id": session_id
                }
            
            # Validate and normalize plate format (Tunisian plates only)
            if plate:
                normalized_plate, is_valid = normalize_tunisian_plate(plate)
                if not is_valid:
                    save_chat_message(session_id, "ai", "Le format de la plaque n'est pas valide. Formats acceptés: '159 تونس 8240' ou '3341323 نت'")
                    return {
                        "status": "chat",
                        "message": "Le format de la plaque n'est pas valide. Veuillez fournir une plaque tunisienne valide.\n\nFormats acceptés:\n- 159 تونس 8240\n- 3341323 نت",
                        "session_id": session_id
                    }
                plate = normalized_plate  # Use the normalized Arabic format
            
            if not client_name:
                return {
                    "status": "chat",
                    "message": "Désolé, vous devez être connecté en tant que client pour passer une commande.",
                    "session_id": session_id
                }
            
            # CREATE THE ORDER NOW (we have all required info)
            result = create_new_order(
                client_name=client_name,
                product_name=det.get('product'),
                quantity=det.get('quantity'),
                plate=plate,
                user_id=user_id,
                plate_source='vision' if detected_plate else 'chat'
            )
            
            if result['success']:
                return {
                    "status": "success",
                    "type": "order_created",
                    "message": f"{decision['response']}\n\n✅ Commande #{result['order_id']} créée avec succès!",
                    "order_id": result['order_id'],
                    "preview": {
                        "client": client_name,
                        "product": det.get('product'),
                        "quantity": det.get('quantity'),
                        "plate": plate
                    },
                    "session_id": session_id
                }
            else:
                return {
                    "status": "error",
                    "type": result.get('error'),
                    "message": f"Erreur: {result.get('reason')}",
                    "session_id": session_id
                }
        
        # ====================================================================
        # DEFAULT: Return chat response
        # ====================================================================
        
        # ====================================================================
        # REGISTER: Create new client
        # ====================================================================
        if intent == "register" and decision["details"].get("new_client_name"):
            res = create_new_client(decision["details"]["new_client_name"])
            if res:
                return {
                    "status": "success",
                    "type": "client_created",
                    "message": f"{decision['response']} (New client registered)",
                    "session_id": session_id
                }
        
        # ====================================================================
        # CHAT: Just return AI response
        # ====================================================================
        return {
            "status": "chat",
            "message": decision['response'],
            "session_id": session_id
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"System error: {str(e)}",
            "session_id": session_id if 'session_id' in locals() else None
        }

@app.post("/confirm-order/{order_id}")
async def api_confirm_order(order_id: int, user_id: Optional[int] = None):
    """
    ✅ FIX #1: Confirm an order (mark as user_confirmed=TRUE)
    Called when user replies "yes" to order confirmation preview
    """
    from src.database import confirm_order
    
    result = confirm_order(order_id, user_id)
    
    if result['success']:
        return {"status": "success", "message": result['message']}
    else:
        raise HTTPException(status_code=400, detail=result['error'])

@app.get("/inventory")
async def get_inventory():
    try:
        from src.database import list_products
        return list_products()
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/inventory")
async def create_product(data: ProductCreate, request: Request):
    """Admin only: Create new product"""
    user_role = request.headers.get("X-User-Role")
    if user_role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    try:
        from src.database import add_product
        id_prod = add_product(data.name, data.stock, data.price)
        if id_prod is None:
            raise HTTPException(status_code=400, detail="Failed to create product")
        return {"status": "success", "id": id_prod}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/inventory/{id_prod}")
async def edit_product(id_prod: int, data: ProductUpdate, request: Request):
    """Admin only: Update product"""
    user_role = request.headers.get("X-User-Role")
    if user_role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    try:
        from src.database import update_product
        update_product(id_prod, data.name, data.stock, data.price)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/inventory/{id_prod}")
async def remove_product(id_prod: int, request: Request):
    """Admin only: Delete product"""
    user_role = request.headers.get("X-User-Role")
    if user_role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    try:
        from src.database import delete_product
        delete_product(id_prod)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/chat-sessions/{user_id}")
async def fetch_chat_sessions(user_id: int):
    try:
        from src.database import get_chat_sessions
        return get_chat_sessions(user_id)
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/chat-messages/{session_id}")
async def fetch_chat_messages(session_id: int):
    try:
        from src.database import get_chat_messages
        return get_chat_messages(session_id)
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/user/{email}")
async def fetch_user(email: str):
    try:
        from src.database import get_user_by_email
        return get_user_by_email(email)
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/update-password")
async def change_password(data: dict):
    try:
        from src.database import update_password
        update_password(data['user_id'], data['new_password'])
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/delete-account")
async def remove_account(data: dict):
    try:
        from src.database import delete_user
        delete_user(data['user_id'])
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
def health():
    # Check if vision and agent were initialized for a more accurate health check
    model_loaded = 'vision' in globals() and 'agent' in globals()
    return {"status": "online", "model_loaded": model_loaded}

# --- Orders API ---
@app.get("/orders")
async def get_orders():
    """Get all orders with client and product details"""
    try:
        from src.database import list_orders
        return list_orders()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders/{order_id}")
async def get_order(order_id: int):
    """Get a specific order by ID"""
    try:
        from src.database import get_order_by_id
        order = get_order_by_id(order_id)
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        return order
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/orders/{order_id}/status")
async def update_order_status_endpoint(order_id: int, data: dict):
    """Update order status"""
    try:
        from src.database import update_order_status
        new_status = data.get("status")
        # Pickup-only flow: awaiting_pickup → picked_up (legacy awaiting allowed)
        valid_statuses = ["awaiting_pickup", "picked_up", "awaiting", "en attente"]
        if new_status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Valid: {valid_statuses}")
        update_order_status(order_id, new_status)
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Clients API ---
@app.get("/clients")
async def get_clients():
    """Get all clients"""
    try:
        from src.database import list_clients_full
        return list_clients_full()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard")
async def get_dashboard():
    """Get comprehensive dashboard metrics"""
    try:
        from src.database import get_dashboard_stats
        return get_dashboard_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register")
async def register_user(user_data: UserRegistration):
    """
    Register a new client user - NO admin approval required.
    Creates user account + client profile immediately.
    """
    try:
        result = register_new_user(
            full_name=user_data.fullName,
            email=user_data.email,
            company=user_data.company,
            password=user_data.password
        )
        
        if result["success"]:
            return {
                "status": "success",
                "message": "Account created successfully! You can now login.",
                "user_id": result["user_id"],
                "client_id": result["client_id"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
async def login(credentials: UserLogin):
    """
    Authenticate user and return profile.
    """
    try:
        result = login_user(
            email=credentials.email,
            password=credentials.password
        )
        
        if result["success"]:
            return {
                "status": "success",
                "user": result["user"]
            }
        else:
            raise HTTPException(status_code=401, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vehicle-arrival")
async def vehicle_arrival(data: dict):
    """
    Called when a vehicle arrives at the gate.
    Vision system detects the plate and sends it here.
    
    Flow:
    1. Receive detected plate from gate camera
    2. Look up awaiting orders for this plate
    3. Notify chatbot about the arrival
    4. Admin confirms pickup through chatbot
    5. Order status changes to 'picked_up'
    """
    try:
        detected_plate = data.get("plate", "").strip()
        
        if not detected_plate:
            return {
                "status": "error",
                "message": "Plate number required"
            }
        
        from src.database import get_awaiting_orders_for_plate
        
        # Normalize plate if needed
        import re
        def normalize_tunisian_plate(plate_str):
            if not plate_str:
                return plate_str
            normalized = plate_str.lower()
            normalized = re.sub(r'\b(tounes|tunisie|tunis|tun)\b', 'تونس', normalized, flags=re.IGNORECASE)
            normalized = re.sub(r'\b(rs|nt)\b', 'نت', normalized, flags=re.IGNORECASE)
            return normalized.strip()
        
        detected_plate = normalize_tunisian_plate(detected_plate)
        
        # Look for awaiting orders with this plate
        result = get_awaiting_orders_for_plate(detected_plate)
        
        if result["found"]:
            order = result["order"]
            return {
                "status": "success",
                "detected_plate": detected_plate,
                "order_found": True,
                "order": {
                    "id": order["idCommande"],
                    "client": order["client_nom"],
                    "product": order["produit_nom"],
                    "quantity": order["quantite"],
                    "created": str(order["dateCommande"]),
                    "plate": order["plaque_vehicule"]
                },
                "message": f"✅ Vehicle detected! Order #{order['idCommande']} for {order['client_nom']} found.\nProduct: {order['produit_nom']} x{order['quantite']}\nReady for pickup confirmation."
            }
        else:
            return {
                "status": "warning",
                "detected_plate": detected_plate,
                "order_found": False,
                "message": f"⚠️ Vehicle plate '{detected_plate}' detected but no awaiting orders found. Please verify the plate or create an order first."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing vehicle arrival: {str(e)}")

@app.post("/confirm-pickup")
async def confirm_pickup_endpoint(data: dict):
    """
    Admin confirms that vehicle has picked up the order.
    Called after admin interacts with chatbot about pickup.
    
    Changes order status from 'awaiting' to 'picked_up'
    """
    try:
        order_id = data.get("order_id")
        admin_user_id = data.get("admin_user_id")
        
        if not order_id:
            return {
                "status": "error",
                "message": "Order ID required"
            }
        
        from src.database import confirm_pickup
        
        result = confirm_pickup(order_id, admin_user_id)
        
        if result["success"]:
            return {
                "status": "success",
                "order_id": order_id,
                "message": result["message"],
                "old_status": result["old_status"],
                "new_status": result["new_status"]
            }
        else:
            return {
                "status": "error",
                "message": result["error"]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error confirming pickup: {str(e)}")

@app.post("/authorize-pickup")
async def authorize_pickup_endpoint(data: dict):
    """
    HIGH PRIORITY FIX #3: Admin authorizes vehicle to proceed to pickup area.
    
    Changes order status from 'detected' or 'awaiting' to 'authorized'.
    Records gate assignment and authorization timestamp.
    
    Expected status flow: awaiting → detected → authorized → in_progress → terminée
    """
    try:
        order_id = data.get("order_id")
        gate_id = data.get("gate_id", "A-01")  # Default gate
        admin_user_id = data.get("admin_user_id")
        
        if not order_id:
            return {
                "status": "error",
                "message": "Order ID required"
            }
        
        # HIGH PRIORITY FIX #4: Admin authentication check
        from src.database import authorize_pickup, get_user_role
        
        # Verify admin privileges
        if admin_user_id:
            user_role = get_user_role(admin_user_id)
            if user_role != 'admin':
                return {
                    "status": "error",
                    "error": "unauthorized",
                    "message": "Only admin users can authorize pickups"
                }
        
        result = authorize_pickup(order_id, gate_id, admin_user_id)
        
        if result["success"]:
            return {
                "status": "success",
                "order_id": order_id,
                "message": result["message"],
                "gate": result["gate"],
                "old_status": result["old_status"],
                "new_status": result["new_status"]
            }
        else:
            return {
                "status": "error",
                "message": result.get("message", result.get("error"))
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error authorizing pickup: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Using 127.0.0.1 to avoid some Windows firewall/binding issues
    uvicorn.run(app, host="127.0.0.1", port=8000)
