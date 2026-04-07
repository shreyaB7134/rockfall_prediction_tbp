"""
Real-time Alert System for Rockfall Prediction
Handles automated notifications when risk exceeds threshold
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/rockfall_alerts.log'),
        logging.StreamHandler()
    ]
)

class RockfallAlertSystem:
    """Alert system for rockfall risk notifications"""
    
    def __init__(self, risk_threshold: float = 0.8):
        self.risk_threshold = risk_threshold
        self.alert_history = []
        self.last_alert_time = {}
        
        # Email configuration (replace with actual credentials)
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'alerts@rockfall-system.com',
            'sender_password': 'your_app_password',  # Use app password for Gmail
            'recipients': ['mine-manager@company.com', 'safety-officer@company.com']
        }
        
        # SMS configuration (Twilio - replace with actual credentials)
        self.sms_config = {
            'account_sid': 'your_twilio_account_sid',
            'auth_token': 'your_twilio_auth_token',
            'twilio_phone': '+1234567890',
            'recipient_phones': ['+0987654321', '+1122334455']
        }
    
    def check_risk_level(self, risk_probability: float, scenario_id: str = "unknown") -> str:
        """Determine risk level based on probability"""
        
        if risk_probability >= self.risk_threshold:
            return "CRITICAL"
        elif risk_probability >= 0.6:
            return "HIGH"
        elif risk_probability >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def send_email_alert(self, risk_probability: float, scenario_data: Dict) -> bool:
        """Send email alert about high rockfall risk"""
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(self.email_config['recipients'])
            msg['Subject'] = f'🚨 CRITICAL: Rockfall Risk Alert - {risk_probability:.1%}'
            
            # Email body
            body = f"""
ROCKFALL RISK ALERT - IMMEDIATE ATTENTION REQUIRED

Risk Level: CRITICAL
Risk Probability: {risk_probability:.1%}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Current Conditions:
- Slope Angle: {scenario_data.get('slope_angle', 'N/A')}°
- Pore Pressure: {scenario_data.get('pore_pressure', 'N/A')}
- Rainfall: {scenario_data.get('rainfall', 'N/A')} mm
- Instability Score: {scenario_data.get('instability_score', 'N/A')}
- Temperature: {scenario_data.get('temperature', 'N/A')}°C
- Humidity: {scenario_data.get('humidity', 'N/A')}%
- Soil Wetness: {scenario_data.get('soil_wetness', 'N/A')}

RECOMMENDED ACTIONS:
1. Immediately evacuate affected areas
2. Deploy ground monitoring teams
3. Alert all personnel in the vicinity
4. Implement emergency response protocols
5. Continuous monitoring until conditions improve

This is an automated alert from the Rockfall Prediction System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email (commented out - requires actual SMTP credentials)
            # server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            # server.starttls()
            # server.login(self.email_config['sender_email'], self.email_config['sender_password'])
            # server.send_message(msg)
            # server.quit()
            
            print(f"EMAIL ALERT SENT (Simulated): Risk {risk_probability:.1%}")
            logging.info(f"Email alert sent for risk level: {risk_probability:.1%}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send email alert: {str(e)}")
            return False
    
    def send_sms_alert(self, risk_probability: float, scenario_data: Dict) -> bool:
        """Send SMS alert about high rockfall risk"""
        
        try:
            message = f"🚨 CRITICAL ROCKFALL RISK: {risk_probability:.1%} - Immediate evacuation required! Check email for details."
            
            # Send SMS using Twilio (commented out - requires actual credentials)
            # client = Client(self.sms_config['account_sid'], self.sms_config['auth_token'])
            # for phone in self.sms_config['recipient_phones']:
            #     client.messages.create(
            #         body=message,
            #         from_=self.sms_config['twilio_phone'],
            #         to=phone
            #     )
            
            print(f"SMS ALERT SENT (Simulated): {message}")
            logging.info(f"SMS alert sent for risk level: {risk_probability:.1%}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send SMS alert: {str(e)}")
            return False
    
    def trigger_alert(self, risk_probability: float, scenario_data: Dict, scenario_id: str = "unknown") -> Dict:
        """Trigger alert system based on risk level"""
        
        risk_level = self.check_risk_level(risk_probability, scenario_id)
        current_time = datetime.now()
        
        alert_result = {
            'timestamp': current_time,
            'scenario_id': scenario_id,
            'risk_probability': risk_probability,
            'risk_level': risk_level,
            'alerts_sent': []
        }
        
        # Check if we should send alerts (avoid spam)
        time_since_last_alert = current_time - self.last_alert_time.get(scenario_id, datetime.min)
        
        if risk_level == "CRITICAL" and time_since_last_alert.total_seconds() > 300:  # 5 minutes cooldown
            print(f"\n🚨 CRITICAL ALERT TRIGGERED 🚨")
            print(f"Risk Probability: {risk_probability:.1%}")
            print(f"Scenario ID: {scenario_id}")
            print(f"Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Send email alert
            if self.send_email_alert(risk_probability, scenario_data):
                alert_result['alerts_sent'].append('email')
            
            # Send SMS alert
            if self.send_sms_alert(risk_probability, scenario_data):
                alert_result['alerts_sent'].append('sms')
            
            # Update last alert time
            self.last_alert_time[scenario_id] = current_time
            
            # Log alert
            logging.warning(f"Critical alert triggered for scenario {scenario_id}: {risk_probability:.1%}")
            
        elif risk_level == "HIGH":
            print(f"\n⚠️ HIGH RISK DETECTED")
            print(f"Risk Probability: {risk_probability:.1%}")
            print(f"Scenario ID: {scenario_id}")
            logging.info(f"High risk detected for scenario {scenario_id}: {risk_probability:.1%}")
        
        else:
            print(f"Risk Level: {risk_level} ({risk_probability:.1%})")
        
        # Store in alert history
        self.alert_history.append(alert_result)
        
        return alert_result
    
    def get_alert_history(self, last_n: int = 10) -> List[Dict]:
        """Get recent alert history"""
        return self.alert_history[-last_n:]
    
    def get_alert_statistics(self) -> Dict:
        """Get alert system statistics"""
        
        if not self.alert_history:
            return {'total_alerts': 0}
        
        total_alerts = len(self.alert_history)
        critical_alerts = len([a for a in self.alert_history if a['risk_level'] == 'CRITICAL'])
        high_alerts = len([a for a in self.alert_history if a['risk_level'] == 'HIGH'])
        
        return {
            'total_alerts': total_alerts,
            'critical_alerts': critical_alerts,
            'high_alerts': high_alerts,
            'critical_percentage': (critical_alerts / total_alerts) * 100 if total_alerts > 0 else 0
        }

def send_alert(risk_level: float, scenario_data: Dict = None) -> None:
    """Simple alert function for quick testing"""
    
    if risk_level > 0.8:
        print("CRITICAL ALERT: High Rockfall Risk Detected!")
        print(f"Risk Level: {risk_level:.1%}")
        if scenario_data:
            print("Current Conditions:")
            for key, value in scenario_data.items():
                print(f"  - {key}: {value}")
        print("RECOMMENDED: Immediate evacuation and safety protocols!")
    else:
        print(f"Risk Level: {risk_level:.1%} - No immediate action required")

def main():
    """Test the alert system"""
    
    print("Testing Rockfall Alert System...")
    
    # Initialize alert system
    alert_system = RockfallAlertSystem(risk_threshold=0.8)
    
    # Test scenarios
    test_scenarios = [
        {
            'risk_probability': 0.95,
            'scenario_data': {
                'slope_angle': 42.0,
                'pore_pressure': 0.85,
                'rainfall': 75.0,
                'instability_score': 4,
                'temperature': 28.0,
                'humidity': 85.0,
                'soil_wetness': 0.7
            },
            'scenario_id': 'test_scenario_1'
        },
        {
            'risk_probability': 0.65,
            'scenario_data': {
                'slope_angle': 35.0,
                'pore_pressure': 0.6,
                'rainfall': 45.0,
                'instability_score': 2,
                'temperature': 22.0,
                'humidity': 60.0,
                'soil_wetness': 0.4
            },
            'scenario_id': 'test_scenario_2'
        },
        {
            'risk_probability': 0.25,
            'scenario_data': {
                'slope_angle': 25.0,
                'pore_pressure': 0.3,
                'rainfall': 15.0,
                'instability_score': 0,
                'temperature': 20.0,
                'humidity': 40.0,
                'soil_wetness': 0.2
            },
            'scenario_id': 'test_scenario_3'
        }
    ]
    
    # Test each scenario
    for scenario in test_scenarios:
        print(f"\n{'='*50}")
        alert_system.trigger_alert(
            scenario['risk_probability'],
            scenario['scenario_data'],
            scenario['scenario_id']
        )
    
    # Display alert statistics
    print(f"\n{'='*50}")
    print("Alert System Statistics:")
    stats = alert_system.get_alert_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nAlert system test completed!")

if __name__ == "__main__":
    main()
