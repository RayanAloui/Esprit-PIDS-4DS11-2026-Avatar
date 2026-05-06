from locust import HttpUser, task, between, TaskSet
import re
import random

# ==========================================================
# CONFIGURATION
# ==========================================================
USERNAME = "RayenAloui"
PASSWORD = "159357"

# Messages for testing the Alia API
SAMPLE_MESSAGES = [
    "Parlez-moi de Vaseline",
    "Quels sont les produits disponibles ?",
    "Analyse du marché actuel",
    "Comment améliorer les ventes ?",
    "Stratégie de recommandation",
]

MODES = ["commercial", "analytics", "recommendation"]

class ALIAUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """
        Login and establish session before running tasks.
        Handles CSRF token extraction and session cookies.
        """
        # Step 1: Get CSRF token from login page
        response = self.client.get("/accounts/login/")
        
        if response.status_code != 200:
            print(f"❌ Cannot reach login page: {response.status_code}")
            return
        
        # Extract CSRF token from the form
        csrf_match = re.search(
            r'name="csrfmiddlewaretoken" value="([^"]+)"',
            response.text
        )
        
        if not csrf_match:
            print("❌ No CSRF token found in login page")
            return
        
        csrf_token = csrf_match.group(1)
        
        # Step 2: Submit login with CSRF token
        login_response = self.client.post(
            "/accounts/login/",
            {
                "username": USERNAME,
                "password": PASSWORD,
                "csrfmiddlewaretoken": csrf_token,
            },
            headers={"Referer": self.host + "/accounts/login/"},
        )
        
        if login_response.status_code in [200, 302]:
            print("✅ Login successful")
        else:
            print(f"❌ Login failed: {login_response.status_code}")
            print(f"   Response: {login_response.text[:200]}")
    
    # ==========================================================
    # HIGH PRIORITY TASKS (weight 3-4)
    # ==========================================================
    
    @task(4)
    def home_page(self):
        """Main landing page - most visited"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Home failed: {response.status_code}")
    
    @task(3)
    def ask_alia(self):
        """Alia AI assistant API - core functionality"""
        csrf = self.client.cookies.get('csrftoken', '')
        payload = {
            "message": random.choice(SAMPLE_MESSAGES),
            "mode": random.choice(MODES),
        }
        
        with self.client.post(
            "/alia-api/ask_alia",
            json=payload,
            headers={
                "X-CSRFToken": csrf,
                "Content-Type": "application/json",
            },
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 403:
                response.failure(f"CSRF failure on ask_alia")
            else:
                response.failure(f"Ask Alia failed: {response.status_code}")
    
    # ==========================================================
    # MEDIUM PRIORITY TASKS (weight 2)
    # ==========================================================
    
    @task(2)
    def simulator_page(self):
        """Training simulator main page"""
        self.client.get("/simulator/")
    
    @task(2)
    def crm_overview(self):
        """CRM overview dashboard"""
        self.client.get("/crm/")
    
    @task(2)
    def analytics_page(self):
        """Analytics dashboard"""
        self.client.get("/analytics/")
    
    @task(2)
    def routes_page(self):
        """Route optimization page"""
        self.client.get("/routes/")

    
    # ==========================================================
    # LOWER PRIORITY TASKS (weight 1)
    # ==========================================================
    
    @task(1)
    def modeling_index(self):
        """Modeling module index"""
        self.client.get("/alia-api/")
    
    @task(1)
    def crm_pharmacies(self):
        """CRM pharmacies listing"""
        self.client.get("/crm/pharmacies/")
    
    @task(1)
    def crm_delegates(self):
        """CRM delegates listing"""
        self.client.get("/crm/delegates/")
    
    @task(1)
    def simulator_qcm_questions(self):
        """Get QCM questions for training"""
        self.client.get("/simulator/qcm/questions/")
    
    @task(1)
    def analytics_data(self):
        """Fetch analytics data"""
        self.client.get("/analytics/data/")
    
    @task(1)
    def routes_optimize(self):
        """Trigger route optimization"""
        self.client.post("/routes/optimize/")
    
    # ==========================================================
    # OCCASIONAL TASKS (weight 1, less frequent)
    # ==========================================================
    
    @task(1)
    def profile_page(self):
        """User profile page"""
        self.client.get("/accounts/profile/")
    
    @task(1)
    def crm_zones(self):
        """CRM zones view"""
        self.client.get("/crm/zones/")
    
    @task(1)
    def crm_predictions(self):
        """CRM predictions view"""
        self.client.get("/crm/predictions/")
    
    @task(1)
    def simulator_dashboard(self):
        """Simulator dashboard data"""
        self.client.get("/simulator/dashboard/")
    
    @task(1)
    def analytics_action_plan(self):
        """Analytics action plan"""
        self.client.get("/analytics/action-plan/")

    
    @task(1)
    def avatar_history(self):
        """Avatar history data"""
        self.client.get("/avatar/history/")
    

    # ==========================================================
    # READ-ONLY ENDPOINTS (safe to test, no side effects)
    # ==========================================================
    
    @task(1)
    def simulator_replay_page(self):
        """Replay page"""
        self.client.get("/simulator/replay/view/")
