#!/usr/bin/env python3
"""
AI Assistant for Leaf Detection System
Provides comprehensive information about leaves with speech functionality
"""

import pyttsx3
import time
import os
from datetime import datetime

class LeafAssistant:
    def __init__(self):
        """Initialize the leaf assistant with speech engine"""
        try:
            self.engine = pyttsx3.init()
            # Configure speech settings
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', 0.9)  # Volume level
            print("✅ Speech engine initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize speech engine: {e}")
            self.engine = None
    
    def speak(self, text):
        """Speak the given text"""
        if self.engine:
            try:
                print(f"🗣️  Speaking: {text}")
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"❌ Speech error: {e}")
        else:
            print(f"📝 Text: {text}")
    
    def get_leaf_information(self, leaf_name):
        """Get comprehensive information about a specific leaf"""
        leaf_data = {
            "Holy Basil": {
                "scientific_name": "Ocimum tenuiflorum",
                "common_names": ["Tulsi", "Holy Basil", "Sacred Basil"],
                "description": "Holy Basil is a sacred plant in Hinduism with strong medicinal properties. It's considered the 'Queen of Herbs' in Ayurveda.",
                "advantages": [
                    "Rich in antioxidants that combat free radicals",
                    "Boosts immunity and helps fight infections",
                    "Reduces stress and anxiety levels",
                    "Helps regulate blood sugar levels",
                    "Supports respiratory health",
                    "Anti-inflammatory properties",
                    "Improves skin health",
                    "Enhances memory and cognitive function",
                    "Natural detoxifier",
                    "Promotes heart health"
                ],
                "disadvantages": [
                    "May lower blood sugar too much in diabetics",
                    "Might slow blood clotting (caution before surgery)",
                    "Possible interactions with blood-thinning medications",
                    "Not recommended in large quantities during pregnancy",
                    "May cause allergic reactions in some people",
                    "Can interfere with certain medications",
                    "Bitter taste may be unpalatable for some",
                    "Should be avoided by people with low blood pressure"
                ],
                "medicinal_uses": [
                    "Treating respiratory disorders like asthma and bronchitis",
                    "Reducing fever and inflammation",
                    "Managing diabetes and cholesterol levels",
                    "Promoting heart health",
                    "Enhancing skin health and treating skin disorders",
                    "Stress relief and anxiety management",
                    "Improving digestion",
                    "Boosting immune system",
                    "Treating headaches and migraines",
                    "Supporting liver function"
                ],
                "growing_tips": [
                    "Prefers warm, tropical climate",
                    "Needs well-drained soil and plenty of sunlight",
                    "Water regularly but avoid waterlogging",
                    "Prune regularly to encourage bushier growth",
                    "Can be grown in pots or directly in the ground",
                    "Plant in spring or early summer",
                    "Requires 6-8 hours of sunlight daily",
                    "Fertilize monthly during growing season",
                    "Protect from frost and cold temperatures",
                    "Harvest leaves regularly to promote new growth"
                ],
                "nutritional_value": [
                    "Rich in Vitamin A, C, and K",
                    "Contains essential minerals like calcium, iron, and zinc",
                    "High in antioxidants and flavonoids",
                    "Good source of dietary fiber",
                    "Contains essential oils with therapeutic properties"
                ]
            },
            "Indian Lilac": {
                "scientific_name": "Azadirachta indica",
                "common_names": ["Neem", "Indian Lilac", "Margosa Tree"],
                "description": "Indian Lilac, commonly known as Neem, is a versatile medicinal tree native to India with powerful antibacterial and antifungal properties.",
                "advantages": [
                    "Powerful natural pesticide and insect repellent",
                    "Promotes healthy skin and treats various skin conditions",
                    "Supports oral health (used in toothpastes and mouthwashes)",
                    "Helps purify blood in traditional medicine",
                    "Has anti-diabetic properties",
                    "Strong antibacterial and antiviral effects",
                    "Natural anti-inflammatory agent",
                    "Supports liver health",
                    "Helps in wound healing",
                    "Effective against fungal infections"
                ],
                "disadvantages": [
                    "Bitter taste makes it unpalatable for some uses",
                    "May cause allergic reactions in sensitive individuals",
                    "Can be toxic in large doses",
                    "Not recommended for infants or pregnant women",
                    "May lower blood sugar too much in diabetics",
                    "Can cause stomach upset in some people",
                    "May interact with certain medications",
                    "Should be used cautiously by people with kidney problems"
                ],
                "medicinal_uses": [
                    "Treating skin conditions like eczema and acne",
                    "Natural insect repellent and pesticide",
                    "Oral care for healthy gums and teeth",
                    "Blood purification in traditional medicine",
                    "Managing diabetes and boosting immunity",
                    "Treating fungal infections",
                    "Wound healing and skin regeneration",
                    "Liver detoxification",
                    "Treating fever and infections",
                    "Supporting digestive health"
                ],
                "growing_tips": [
                    "Thrives in hot, dry climates",
                    "Drought-resistant once established",
                    "Prefers well-drained soil",
                    "Can grow up to 15-20 meters tall",
                    "Fast-growing tree with many uses",
                    "Plant in full sunlight",
                    "Tolerates poor soil conditions",
                    "Requires minimal maintenance",
                    "Can be grown from seeds or cuttings",
                    "Prune regularly to maintain shape"
                ],
                "nutritional_value": [
                    "Rich in antioxidants and flavonoids",
                    "Contains essential fatty acids",
                    "Good source of vitamin C and E",
                    "Contains minerals like calcium, phosphorus, and iron",
                    "High in protein and amino acids"
                ]
            },
            "Polyalthia Longifolia": {
                "scientific_name": "Polyalthia longifolia",
                "common_names": ["Mast Tree", "False Ashoka", "Indian Fir Tree"],
                "description": "Polyalthia Longifolia is an evergreen tree native to India, primarily ornamental but has some medicinal uses in traditional medicine.",
                "advantages": [
                    "Excellent noise pollution reducer (dense foliage)",
                    "Used in traditional medicine for fever and skin diseases",
                    "Provides good shade with minimal root damage",
                    "Wind-resistant and low-maintenance tree",
                    "Used in Ayurveda for its anti-inflammatory properties",
                    "Ornamental value for landscaping",
                    "Helps in soil conservation",
                    "Provides habitat for birds and wildlife",
                    "Drought-tolerant once established",
                    "Long lifespan and durability"
                ],
                "disadvantages": [
                    "Limited medicinal uses compared to other plants",
                    "Not as well-studied for therapeutic benefits",
                    "Some people may be allergic to its pollen",
                    "Large size makes it unsuitable for small gardens",
                    "Can be expensive to purchase and plant",
                    "Requires significant space to grow properly",
                    "May drop leaves and flowers creating mess",
                    "Roots can damage nearby structures if not managed"
                ],
                "medicinal_uses": [
                    "Fever reduction in traditional medicine",
                    "Treating skin conditions and wounds",
                    "Mild anti-inflammatory properties",
                    "Sometimes used for digestive issues",
                    "Occasional use in traditional pain relief",
                    "Supporting respiratory health",
                    "Helping with stress and anxiety",
                    "Promoting wound healing",
                    "Supporting immune system",
                    "Traditional use for heart health"
                ],
                "growing_tips": [
                    "Prefers tropical to subtropical climates",
                    "Needs space to grow (can reach 30m tall)",
                    "Low maintenance once established",
                    "Tolerates various soil types",
                    "Often planted as avenue trees for shade",
                    "Plant in well-drained soil",
                    "Requires full sunlight",
                    "Water regularly during establishment",
                    "Prune to maintain desired shape",
                    "Protect young trees from strong winds"
                ],
                "nutritional_value": [
                    "Contains various bioactive compounds",
                    "Rich in antioxidants",
                    "Contains essential oils",
                    "Good source of flavonoids",
                    "Contains minerals and vitamins"
                ]
            }
        }
        
        return leaf_data.get(leaf_name, None)
    
    def speak_leaf_info(self, leaf_name):
        """Speak comprehensive information about a leaf"""
        leaf_info = self.get_leaf_information(leaf_name)
        
        if not leaf_info:
            self.speak(f"Sorry, I don't have information about {leaf_name}")
            return
        
        # Introduction
        intro = f"Let me tell you about {leaf_name}, also known as {', '.join(leaf_info['common_names'][:2])}. "
        intro += f"Its scientific name is {leaf_info['scientific_name']}. "
        intro += leaf_info['description']
        self.speak(intro)
        time.sleep(1)
        
        # Advantages
        self.speak(f"Here are the main advantages of {leaf_name}:")
        for i, advantage in enumerate(leaf_info['advantages'][:5], 1):
            self.speak(f"{i}. {advantage}")
            time.sleep(0.5)
        
        time.sleep(1)
        
        # Disadvantages
        self.speak(f"Now, let me mention some disadvantages of {leaf_name}:")
        for i, disadvantage in enumerate(leaf_info['disadvantages'][:4], 1):
            self.speak(f"{i}. {disadvantage}")
            time.sleep(0.5)
        
        time.sleep(1)
        
        # Medicinal uses
        self.speak(f"Here are the main medicinal uses of {leaf_name}:")
        for i, use in enumerate(leaf_info['medicinal_uses'][:5], 1):
            self.speak(f"{i}. {use}")
            time.sleep(0.5)
        
        time.sleep(1)
        
        # Growing tips
        self.speak(f"Finally, here are some growing tips for {leaf_name}:")
        for i, tip in enumerate(leaf_info['growing_tips'][:5], 1):
            self.speak(f"{i}. {tip}")
            time.sleep(0.5)
        
        self.speak(f"That's all about {leaf_name}. Thank you for listening!")
    
    def speak_welcome_message(self):
        """Speak welcome message"""
        welcome = "Hello! I'm your AI Leaf Assistant. I can provide detailed information about Holy Basil, Indian Lilac, and Polyalthia Longifolia. "
        welcome += "Just ask me about any of these leaves and I'll tell you their advantages, disadvantages, medicinal uses, and growing tips. "
        welcome += "You can also ask me to speak about all leaves at once."
        self.speak(welcome)
    
    def speak_all_leaves_summary(self):
        """Speak a summary of all leaves"""
        self.speak("Let me give you a quick overview of all three leaves in our system.")
        time.sleep(1)
        
        # Holy Basil summary
        self.speak("Holy Basil, or Tulsi, is a sacred plant with strong medicinal properties. It's excellent for boosting immunity, reducing stress, and supporting respiratory health.")
        time.sleep(1)
        
        # Indian Lilac summary
        self.speak("Indian Lilac, or Neem, is a powerful medicinal tree with antibacterial and antifungal properties. It's great for skin health, oral care, and natural pest control.")
        time.sleep(1)
        
        # Polyalthia Longifolia summary
        self.speak("Polyalthia Longifolia, or Mast Tree, is primarily an ornamental tree with some medicinal uses. It's excellent for noise reduction and provides good shade.")
        time.sleep(1)
        
        self.speak("Each leaf has unique properties and uses. Would you like me to provide detailed information about any specific leaf?")
    
    def interactive_mode(self):
        """Run interactive mode for user queries"""
        print("\n🌿 AI Leaf Assistant - Interactive Mode")
        print("=" * 50)
        print("Available commands:")
        print("- 'holy basil' or 'tulsi' - Information about Holy Basil")
        print("- 'indian lilac' or 'neem' - Information about Indian Lilac")
        print("- 'polyalthia' or 'mast tree' - Information about Polyalthia Longifolia")
        print("- 'all' - Summary of all leaves")
        print("- 'welcome' - Welcome message")
        print("- 'quit' or 'exit' - Exit the assistant")
        print("=" * 50)
        
        self.speak_welcome_message()
        
        while True:
            try:
                user_input = input("\n🎤 What would you like to know about? ").lower().strip()
                
                if user_input in ['quit', 'exit', 'bye']:
                    self.speak("Thank you for using the AI Leaf Assistant. Goodbye!")
                    break
                
                elif user_input in ['holy basil', 'tulsi', 'basil']:
                    self.speak_leaf_info("Holy Basil")
                
                elif user_input in ['indian lilac', 'neem', 'lilac']:
                    self.speak_leaf_info("Indian Lilac")
                
                elif user_input in ['polyalthia', 'mast tree', 'polyalthia longifolia']:
                    self.speak_leaf_info("Polyalthia Longifolia")
                
                elif user_input in ['all', 'summary', 'overview']:
                    self.speak_all_leaves_summary()
                
                elif user_input in ['welcome', 'hello', 'hi']:
                    self.speak_welcome_message()
                
                else:
                    self.speak("I'm sorry, I didn't understand that. Please ask about Holy Basil, Indian Lilac, or Polyalthia Longifolia.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function to run the assistant"""
    print("🌿 Initializing AI Leaf Assistant...")
    
    assistant = LeafAssistant()
    
    if assistant.engine:
        print("✅ Assistant ready with speech capabilities!")
    else:
        print("⚠️  Assistant running in text-only mode")
    
    # Run interactive mode
    assistant.interactive_mode()

if __name__ == "__main__":
    main()