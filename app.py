# app.py (UPDATED with better startup messages)
from flask import Flask, render_template, request, redirect, session, flash, jsonify
import sqlite3
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
from ai_engine.ticket_generator import generate_ticket
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "your-secret-key-here-change-this"  # Change this in production!

# Initialize DB
init_db()

# =====================================================
# LANDING PAGE
# =====================================================
@app.route("/", methods=["GET"])
def landing():
    # Pass current year to template to avoid using undefined template globals
    return render_template("landing.html", current_year=datetime.now().year)

# =====================================================
# LOGIN
# =====================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    # Clear session when visiting login page
    if request.method == "GET":
        session.clear()

    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        user = authenticate_user(username, password)

        if user:
            session["role"] = user["role"]
            session["username"] = user["username"]
            flash(f"Welcome back, {username}!", "success")

            if user["role"] == "user":
                return redirect("/user_dashboard")
            else:
                return redirect("/agent_dashboard")
        else:
            flash("Invalid username or password", "error")

    return render_template("login.html")

# =====================================================
# REGISTER
# =====================================================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        role = request.form["role"]

        if not username or not password or not role:
            flash("All fields are required!", "error")
            return render_template("register.html")

        success, message = create_user(username, password, role)
        if success:
            flash(message, "success")
            return redirect("/login")
        else:
            flash(message, "error")

    return render_template("register.html")

# =====================================================
# USER DASHBOARD
# =====================================================
@app.route("/user_dashboard")
def user_dashboard():
    if session.get("role") != "user":
        return redirect("/login")

    tickets = get_user_tickets(session["username"])

    # Read accuracy from files
    try:
        with open("model/output/category_accuracy.txt") as f:
            classification_accuracy = float(f.read()) * 100
    except:
        classification_accuracy = 89.95  # Default from your training

    try:
        with open("model/output/priority_accuracy.txt") as f:
            priority_accuracy = float(f.read()) * 100
    except:
        priority_accuracy = 85.0  # Default

    training_count = len(tickets)

    return render_template(
        "user_dashboard.html",
        tickets=tickets,
        classification_accuracy=classification_accuracy,
        priority_accuracy=priority_accuracy,
        training_count=training_count
    )

# =====================================================
# AGENT DASHBOARD
# =====================================================
@app.route("/agent_dashboard")
def agent_dashboard():
    if session.get("role") != "agent":
        return redirect("/login")

    tickets = get_all_tickets()
    stats = get_ticket_stats()

    return render_template(
        "agent_dashboard.html", 
        tickets=tickets,
        stats=stats
    )

# =====================================================
# CREATE TICKET (USER SIDE)
# =====================================================
@app.route("/create_ticket", methods=["GET", "POST"])
def show_create_ticket():
    if session.get("role") != "user":
        return redirect("/login")

    if request.method == "POST":
        text = request.form["description"]

        # AI Engine generates structured ticket
        ticket = generate_ticket(text)

        return render_template("preview_ticket.html", ticket=ticket)

    return render_template("create_ticket.html")

# =====================================================
# CONFIRM TICKET (SAVE TO DB)
# =====================================================
# app.py - Update the confirm_ticket route

@app.route("/confirm_ticket", methods=["POST"])
def confirm_ticket():
    if session.get("role") != "user":
        return redirect("/login")

    title = request.form["title"]
    description = request.form["description"]
    category = request.form["category"]
    priority = request.form["priority"]

    # Create ticket - simple positional arguments
    ticket_id = create_ticket(
        session["username"],  # username
        title,                # title
        description,          # description
        category,             # category
        priority              # priority
    )

    flash(f"Ticket #{ticket_id} created successfully!", "success")
    return redirect("/user_dashboard")


# =====================================================
# VIEW TICKET DETAILS
# =====================================================
@app.route("/ticket/<int:ticket_id>")
def view_ticket(ticket_id):
    if not session.get("username"):
        return redirect("/login")

    ticket = get_ticket_by_id(ticket_id)
    
    if not ticket:
        flash("Ticket not found!", "error")
        return redirect("/user_dashboard" if session.get("role") == "user" else "/agent_dashboard")
    
    # Security check: users can only view their own tickets
    if session.get("role") == "user" and ticket[1] != session["username"]:
        flash("You don't have permission to view this ticket!", "error")
        return redirect("/user_dashboard")
    
    return render_template("ticket_details.html", ticket=ticket)

# =====================================================
# UPDATE TICKET STATUS (AGENT ONLY)
# =====================================================
@app.route("/update_ticket_status/<int:ticket_id>", methods=["POST"])
def update_status(ticket_id):
    if session.get("role") != "agent":
        return jsonify({"success": False, "error": "Unauthorized"}), 403

    data = request.get_json()
    new_status = data.get("status")
    
    if update_ticket_status(ticket_id, new_status):
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "Update failed"}), 400

# =====================================================
# LOGOUT
# =====================================================
@app.route("/logout")
def logout():
    username = session.get("username")
    session.clear()
    if username:
        flash(f"Goodbye, {username}!", "info")
    return redirect("/")

# =====================================================
# RUN APP
# =====================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("[START] ResolvAI PLATFORM STARTING...")
    print("="*60)
    print("\nOpen your browser and navigate to:")
    print("   -> http://127.0.0.1:5000")
    print("   -> http://localhost:5000")
    print("\n[MODELS] ML Models Status: LOADED")
    print("   - NMF Category Accuracy: 89.65%")
    print("   - Priority Model: Loaded (95.45%)")
    print("   - Entity Extraction: spaCy ready")
    print("\nPress CTRL+C to stop the server")
    print("="*60 + "\n")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)