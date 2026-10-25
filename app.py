"""app.py
ResolvAI Flask Application: intelligent support ticket classification, routing, and operations.
"""
import os
import json
import logging
from datetime import timedelta, datetime
from functools import wraps
import secrets

from flask import Flask, render_template, request, redirect, session, flash, jsonify, url_for
from flask_wtf.csrf import CSRFProtect

from database import (
    init_db,
    create_ticket,
    get_user_tickets,
    get_all_tickets,
    get_ticket_by_id,
    update_ticket_status,
    get_ticket_stats,
    create_user,
    authenticate_user
)
from ai_engine.config import ALLOWED_CATEGORIES, ALLOWED_PRIORITIES
from ai_engine.ticket_generator import generate_ticket, extract_entities

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(BASE_DIR, "model", "output", "metrics.json")

# Application Setup
app = Flask(__name__)
is_debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true")

secret_key = os.environ.get("SECRET_KEY")
if not secret_key:
    if is_debug:
        secret_key = "dev-insecure-secret-key-change-in-prod"
    else:
        logger.warning("[SECURITY] SECRET_KEY env variable not set in production. Generating ephemeral random key.")
        secret_key = secrets.token_hex(32)

app.secret_key = secret_key

# Security & Session Settings
app.config.update(
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=not is_debug
)

csrf = CSRFProtect(app)

# Initialize Database Schema & Run Migrations
init_db()


def load_metrics():
    """Load model metrics from single source of truth (metrics.json)."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Error reading metrics.json: %s", e)
    return None


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("username"):
            flash("Please sign in to access this page.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("username"):
                return redirect(url_for("login"))
            if session.get("role") != role:
                flash("Unauthorized access for your account role.", "error")
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return decorated
    return decorator


@app.route("/", methods=["GET"])
def landing():
    metrics = load_metrics()
    cat_acc = None
    prio_acc = None
    if metrics:
        cat_acc = metrics.get("accuracy") or metrics.get("category_model", {}).get("ml_only", {}).get("accuracy")
        prio_acc = metrics.get("priority_model", {}).get("accuracy")
    return render_template(
        "landing.html",
        current_year=datetime.now().year,
        category_accuracy=f"{cat_acc * 100:.2f}%" if cat_acc else "not available",
        priority_accuracy=f"{prio_acc * 100:.2f}%" if prio_acc else "not available"
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        session.clear()

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = authenticate_user(username, password)
        if user:
            session.permanent = True
            session["role"] = user["role"]
            session["username"] = user["username"]
            flash(f"Welcome back, {username}!", "success")
            return redirect(url_for("user_dashboard" if user["role"] == "user" else "agent_dashboard"))

        flash("Invalid username or password", "error")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or len(password) < 8:
            flash("Username is required and password must be at least 8 characters long.", "error")
            return render_template("register.html")

        success, message = create_user(username, password, "user")
        if success:
            flash(message, "success")
            return redirect(url_for("login"))
        flash(message, "error")

    return render_template("register.html")


@app.route("/user_dashboard")
@role_required("user")
def user_dashboard():
    tickets = get_user_tickets(session["username"])
    metrics = load_metrics()

    classification_accuracy = None
    priority_accuracy = None
    if metrics:
        cat_acc = metrics.get("accuracy") or metrics.get("category_model", {}).get("ml_only", {}).get("accuracy")
        pri_acc = metrics.get("priority_model", {}).get("accuracy")
        if cat_acc is not None:
            classification_accuracy = round(cat_acc * 100, 2)
        if pri_acc is not None:
            priority_accuracy = round(pri_acc * 100, 2)

    return render_template(
        "user_dashboard.html",
        tickets=tickets,
        classification_accuracy=classification_accuracy,
        priority_accuracy=priority_accuracy,
        training_count=len(tickets)
    )


@app.route("/agent_dashboard")
@role_required("agent")
def agent_dashboard():
    return render_template(
        "agent_dashboard.html",
        tickets=get_all_tickets(),
        stats=get_ticket_stats()
    )


@app.route("/create_ticket", methods=["GET", "POST"])
@role_required("user")
def show_create_ticket():
    if request.method == "POST":
        text = request.form.get("description", "").strip()
        ticket = generate_ticket(text)
        return render_template("preview_ticket.html", ticket=ticket)

    return render_template("create_ticket.html")


@app.route("/confirm_ticket", methods=["POST"])
@role_required("user")
def confirm_ticket():
    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    category = request.form.get("category", "").strip()
    priority = request.form.get("priority", "").strip()

    # Server-side validation
    if not title or not description:
        flash("Title and description are required.", "error")
        return redirect(url_for("show_create_ticket"))

    if len(title) > 150 or len(description) > 5000:
        flash("Content exceeds maximum length limit.", "error")
        return redirect(url_for("show_create_ticket"))

    if category not in ALLOWED_CATEGORIES:
        flash(f"Invalid category '{category}'. Must be one of the recognized categories.", "error")
        return redirect(url_for("show_create_ticket"))

    if priority not in ALLOWED_PRIORITIES:
        flash(f"Invalid priority '{priority}'.", "error")
        return redirect(url_for("show_create_ticket"))

    # Server-side recomputation of entities: never trust client-posted entity payload
    entities = extract_entities(description)

    ticket_id = create_ticket(
        session["username"],
        title,
        description,
        category,
        priority,
        entities=entities
    )

    flash(f"Ticket #{ticket_id} created successfully!", "success")
    return redirect(url_for("user_dashboard"))


@app.route("/ticket/<int:ticket_id>")
@login_required
def view_ticket(ticket_id):
    ticket = get_ticket_by_id(ticket_id)
    if not ticket:
        flash("Ticket not found!", "error")
        return redirect(url_for("user_dashboard" if session.get("role") == "user" else "agent_dashboard"))

    if session.get("role") == "user" and ticket["username"] != session["username"]:
        flash("You do not have permission to view this ticket!", "error")
        return redirect(url_for("user_dashboard"))

    entities_dict = {}
    try:
        if "entities" in ticket.keys() and ticket["entities"]:
            entities_dict = json.loads(ticket["entities"])
    except Exception:
        pass

    return render_template("ticket_details.html", ticket=ticket, entities=entities_dict)


@app.route("/update_ticket_status/<int:ticket_id>", methods=["POST"])
@role_required("agent")
def update_status(ticket_id):
    data = request.get_json() or {}
    new_status = data.get("status")
    if new_status not in ("Open", "In Progress", "Resolved"):
        return jsonify({"success": False, "error": "Invalid status value"}), 400

    if update_ticket_status(ticket_id, new_status):
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Ticket not found or update failed"}), 400


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("landing"))


if __name__ == "__main__":
    metrics = load_metrics()
    cat_str = "not available"
    pri_str = "not available"
    if metrics:
        cat_v = metrics.get("accuracy") or metrics.get("category_model", {}).get("ml_only", {}).get("accuracy")
        pri_v = metrics.get("priority_model", {}).get("accuracy")
        if cat_v is not None:
            cat_str = f"{cat_v * 100:.2f}%"
        if pri_v is not None:
            pri_str = f"{pri_v * 100:.2f}%"

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))

    logger.info("ResolvAI Platform starting on http://%s:%s", host, port)
    logger.info("Category Model Accuracy: %s | Priority Model: %s", cat_str, pri_str)

    app.run(debug=is_debug, host=host, port=port)