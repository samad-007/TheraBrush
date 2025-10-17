import os
import json
# Using v1beta as shown in the official example
import google.generativeai as genai
from performance_metrics import track_ai_performance

class ChatGPTAdvisor:
    def __init__(self, api_key=None):
        """Initialize the Gemini advisor (despite the class name)"""
        # Use environment variable if no API key provided
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        
        # Log API key status (but never log the key itself)
        if self.api_key:
            print("Gemini API key detected")
            # Configure the Gemini API - using v1beta as shown in the official example
            genai.configure(api_key=self.api_key)
        else:
            print("No Gemini API key found - will use rule-based suggestions")
    
    @track_ai_performance
    def get_art_suggestions(self, drawing_type, emotion, previous_suggestions=None):
        """Get art suggestions based on the drawing and emotion"""
        if not self.api_key:
            # Fallback to rule-based suggestions if no API key
            return self.rule_based_suggestions(drawing_type, emotion)
        
        try:
            # Context for the Gemini prompt
            drawing_context = f"a sketch that appears to be a {drawing_type}"
            emotion_context = f"The user is feeling {emotion}"
            
            # Include previous suggestions if available
            history_context = ""
            if previous_suggestions:
                history_context = f"Previously, I suggested: {previous_suggestions}"
            
            # Craft the prompt for Gemini
            prompt = f"""
            As an art therapy assistant, I'm helping someone with {drawing_context}. {emotion_context}.
            {history_context}
            
            Please provide a single paragraph of therapeutic advice that includes:
            1. One tool suggestion (pencil, brush, spray, or eraser)
            2. Two color suggestions that would complement their emotional state
            3. A specific technique or element they could add to their drawing
            
            Keep your response conversational, supportive, and directly focused on the art therapy benefits.
            """
            
            # Configure the model
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 150,
            }
            
            # Create the model - using gemini-2.0-flash as shown in the official example
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=generation_config
            )
            
            # Generate the response using the v1beta API format
            response = model.generate_content(prompt)
            
            if response and hasattr(response, 'text'):
                suggestion = response.text.strip()
                # Include the text length in the return for better metrics tracking
                return {"suggestion": suggestion, "source": "gemini", "text_length": len(suggestion)}
            else:
                print("Empty response from Gemini API")
                return self.rule_based_suggestions(drawing_type, emotion)
                
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            # If the specific model is not found, fall back to gemini-pro
            if "is not found" in str(e):
                print("Falling back to gemini-pro model")
                try:
                    model = genai.GenerativeModel(
                        model_name="gemini-pro", 
                        generation_config=generation_config
                    )
                    response = model.generate_content(prompt)
                    if response and hasattr(response, 'text'):
                        suggestion = response.text.strip()
                        # Include the text length in the return for better metrics tracking
                        return {"suggestion": suggestion, "source": "gemini (fallback)", "text_length": len(suggestion)}
                except Exception as fallback_error:
                    print(f"Fallback also failed: {str(fallback_error)}")
                
            return self.rule_based_suggestions(drawing_type, emotion)
    
    @track_ai_performance
    def get_detailed_art_suggestions(self, drawing_context, emotion, previous_suggestions=None):
        """Get detailed art suggestions based on more complete drawing analysis"""
        if not self.api_key:
            # Fallback to rule-based suggestions if no API key
            return self.rule_based_suggestions(drawing_context["type"], emotion)
        
        try:
            # Parse the drawing context for a more detailed prompt
            drawing_type = drawing_context["type"]
            colors = drawing_context.get("colors", [])
            density_map = drawing_context.get("density_map", [])
            details = drawing_context.get("details", {})
            
            # Create a more detailed context description
            color_description = "No colors detected"
            if colors:
                color_description = f"Using colors: {', '.join(colors)}"
            
            # Determine drawing composition based on density map
            composition = "evenly distributed"
            if density_map:
                # Simplified composition analysis
                if density_map[4] > sum(density_map) / 9 * 2:  # Center is much denser
                    composition = "centered"
                elif density_map[0] + density_map[1] + density_map[3] > sum(density_map) / 2:
                    composition = "top-left heavy"
                elif density_map[1] + density_map[2] + density_map[5] > sum(density_map) / 2:
                    composition = "top-right heavy"
                elif density_map[0] + density_map[3] + density_map[6] > sum(density_map) / 2:
                    composition = "left-aligned"
                elif density_map[2] + density_map[5] + density_map[8] > sum(density_map) / 2:
                    composition = "right-aligned"
            
            # Extract additional details for context
            aspect_ratio = details.get("aspect_ratio", 1.0)
            circularity = details.get("circularity", 0.0)
            pixel_density = details.get("pixel_density", 0.0)
            
            drawing_details = f"The drawing appears to be a {drawing_type}, {color_description}, with a {composition} composition."
            
            if drawing_type == "face":
                drawing_details += " I can see what looks like a face with facial features."
            elif drawing_type == "landscape":
                drawing_details += " It has a horizontal layout suggesting a landscape or scenery."
            elif drawing_type == "sun":
                drawing_details += " It has a circular shape that resembles a sun or similar round object."
            elif drawing_type == "dense_pattern":
                drawing_details += " It has a dense pattern with multiple elements close together."
            elif drawing_type == "multiple_objects":
                drawing_details += " It contains multiple distinct objects or elements."
                
            # Include level of detail
            if pixel_density < 0.05:
                drawing_details += " The drawing is very minimal with few lines."
            elif pixel_density > 0.3:
                drawing_details += " The drawing is quite detailed with many lines or filled areas."
                
            # Emotional context
            emotion_context = f"The user is currently feeling {emotion}."
            
            # Previous suggestions context
            history_context = ""
            if previous_suggestions:
                history_context = f"Previously, I suggested: {previous_suggestions}"
            
            # Craft the prompt for Gemini with detailed context
            prompt = f"""
            I'm acting as an art therapy assistant. I'm helping someone with their drawing. {drawing_details} {emotion_context}
            {history_context}
            
            Please provide a detailed therapeutic suggestion that:
            1. Acknowledges what they've drawn so far
            2. Suggests a specific drawing technique that would work well with their current artwork (like hatching, contour lines, stippling, etc.)
            3. Recommends a specific element they could add that would complement what they've already drawn
            4. Suggests a tool (pencil, brush, spray, or eraser) that would work well for these additions
            5. Recommends 1-2 colors that would enhance their drawing AND complement their current emotional state ({emotion})
            
            Your suggestion should be specific to what they've already drawn, helping them build upon it rather than starting something new.
            Focus on therapeutic benefits like self-expression, emotional release, or mindfulness.
            Keep your response conversational, supportive, and concise (about 4-5 sentences).
            """
            
            # Configure the model
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 200,
            }
            
            # Create the model - using gemini-2.0-flash as shown in the official example
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=generation_config
            )
            
            # Generate the response using the v1beta API format
            response = model.generate_content(prompt)
            
            if response and hasattr(response, 'text'):
                suggestion = response.text.strip()
                # Include the text length in the return for better metrics tracking
                return {"suggestion": suggestion, "source": "gemini", "text_length": len(suggestion)}
            else:
                print("Empty response from Gemini API")
                return self.rule_based_suggestions(drawing_type, emotion)
                
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            # If the specific model is not found, fall back to gemini-pro
            if "is not found" in str(e):
                print("Falling back to gemini-pro model")
                try:
                    model = genai.GenerativeModel(
                        model_name="gemini-pro", 
                        generation_config=generation_config
                    )
                    response = model.generate_content(prompt)
                    if response and hasattr(response, 'text'):
                        suggestion = response.text.strip()
                        # Include the text length in the return for better metrics tracking
                        return {"suggestion": suggestion, "source": "gemini (fallback)", "text_length": len(suggestion)}
                except Exception as fallback_error:
                    print(f"Fallback also failed: {str(fallback_error)}")

            return self.rule_based_suggestions(drawing_context["type"], emotion)
    
    def rule_based_suggestions(self, drawing_type, emotion):
        """Provide rule-based suggestions when API is not available"""
        # Map drawing types to appropriate tools and techniques
        drawing_map = {
            "face": {
                "tools": ["pencil", "brush"],
                "techniques": ["add more facial expressions", "focus on the eyes", "add shading for depth"]
            },
            "landscape": {
                "tools": ["brush", "spray"],
                "techniques": ["add a horizon line", "create depth with layers", "add trees or mountains"]
            },
            "flower": {
                "tools": ["brush", "pencil"],
                "techniques": ["add more petals", "create a garden scene", "add color variations"]
            },
            "sun": {
                "tools": ["brush", "spray"],
                "techniques": ["add rays of light", "create a sunset scene", "add warm colors"]
            },
            "tree": {
                "tools": ["brush", "pencil"],
                "techniques": ["add more branches", "create a forest", "add seasonal elements"]
            },
            "cloud": {
                "tools": ["spray", "brush"],
                "techniques": ["add more clouds", "create a sky scene", "add weather elements"]
            },
            "abstract": {
                "tools": ["brush", "spray"],
                "techniques": ["experiment with patterns", "add more shapes", "play with textures"]
            },
            "mountain": {
                "tools": ["pencil", "brush"],
                "techniques": ["add snow caps", "create a range of mountains", "add a path or river"]
            },
            "circle": {
                "tools": ["brush", "pencil"],
                "techniques": ["add concentric patterns", "create a mandala design", "add radial elements"]
            },
            "sketch": {
                "tools": ["pencil", "eraser"],
                "techniques": ["add more details", "enhance outlines", "add hatching for texture"]
            },
            "dense_pattern": {
                "tools": ["pencil", "eraser"],
                "techniques": ["add negative space", "create focal points", "vary line weights"]
            },
            "multiple_objects": {
                "tools": ["pencil", "brush"],
                "techniques": ["connect elements with lines", "add a background", "create unity through color"]
            },
            "horizon": {
                "tools": ["brush", "spray"],
                "techniques": ["add elements in the foreground", "create a sunset/sunrise", "add clouds or birds"]
            },
            "empty": {
                "tools": ["pencil", "brush"],
                "techniques": ["start with simple shapes", "try a spiral pattern", "begin with a central element"]
            }
        }
        
        # Map emotions to color palettes and therapeutic approaches
        emotion_map = {
            "happy": {
                "colors": ["#FFD700", "#FF69B4", "#87CEFA"],
                "approach": "enhance the joyful elements"
            },
            "sad": {
                "colors": ["#4682B4", "#9370DB", "#B0C4DE"],
                "approach": "add elements of hope or light"
            },
            "angry": {
                "colors": ["#8B4513", "#D2691E", "#F4A460"],
                "approach": "create balance with calming elements"
            },
            "neutral": {
                "colors": ["#A9A9A9", "#D3D3D3", "#778899"],
                "approach": "add elements that spark interest"
            },
            "surprised": {
                "colors": ["#FFD700", "#00FFFF", "#FF6347"],
                "approach": "explore the unexpected elements"
            },
            "fear": {
                "colors": ["#483D8B", "#4682B4", "#87CEFA"],
                "approach": "introduce comforting elements"
            },
            "disgust": {
                "colors": ["#556B2F", "#8FBC8F", "#2E8B57"],
                "approach": "transform the unpleasant into something interesting"
            }
        }
        
        # Get the appropriate mappings or use defaults
        drawing_info = drawing_map.get(drawing_type, drawing_map["abstract"])
        emotion_info = emotion_map.get(emotion, emotion_map["neutral"])
        
        # Select a tool, colors, and technique
        tool = drawing_info["tools"][0]
        colors = emotion_info["colors"][:2]
        technique = drawing_info["techniques"][0]
        approach = emotion_info["approach"]
        
         # Generate more detailed suggestions based on the drawing type
        if drawing_type == "face":
            suggestion = f"I notice you're drawing a face while feeling {emotion}. Try using the {tool} tool to add more expressive details to the eyes and mouth - they're the windows to emotion! The colors {colors[0]} and {colors[1]} would work well to {approach}. Try adding some hatching for shading around the features to create more depth and dimension."
        elif drawing_type == "landscape":
            suggestion = f"Your landscape drawing reflects your {emotion} mood. Consider using the {tool} tool to add layers of depth to your horizon. Adding some {colors[0]} and {colors[1]} elements would help {approach}. Try the technique of atmospheric perspective - making distant objects lighter and less detailed - to create a sense of space that can feel expansive and freeing."
        elif drawing_type == "circle" or drawing_type == "sun":
            suggestion = f"I see you've created a circular shape while feeling {emotion}. The {tool} tool would be perfect for adding radiating lines or patterns from the center. Using {colors[0]} and {colors[1]} can help {approach}. Try creating a mandala-like pattern, which can be very meditative and help center your thoughts."
        elif drawing_type == "dense_pattern":
            suggestion = f"You've created an intricate pattern while feeling {emotion}. The {tool} tool can help you create some negative space to balance the density. Try incorporating {colors[0]} and {colors[1]} to {approach}. Varying the line weights and spacing in your pattern can create visual rhythm that mirrors emotional rhythms."
        elif drawing_type == "multiple_objects":
            suggestion = f"I see you've drawn several distinct elements while feeling {emotion}. Using the {tool} tool, try creating connecting lines or shapes between them. Adding touches of {colors[0]} and {colors[1]} can help {approach}. Consider how these elements relate to each other - are they supporting, contrasting, or transforming? This reflection can mirror how different emotions interact within us."
        else:
            suggestion = f"I notice you're feeling {emotion} and drawing what looks like a {drawing_type}. Try using the {tool} tool with {colors[0]} and {colors[1]} to {technique}. This can help you {approach} in your artwork, creating a more therapeutic experience."
        
        return {"suggestion": suggestion, "source": "rule-based"}