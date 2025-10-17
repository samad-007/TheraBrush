import os
import json
from performance_metrics import track_ai_performance

# Try to import Gemini API, but make it optional
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google Generative AI library not installed. Install with: pip install google-generativeai")

class ChatGPTAdvisor:
    def __init__(self, api_key=None):
        """Initialize the AI advisor (using Gemini API)"""
        # Use environment variable if no API key provided
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.api_available = GEMINI_AVAILABLE and bool(self.api_key)
        
        # Log API key status (but never log the key itself)
        if not GEMINI_AVAILABLE:
            print("Gemini API library not available - will use rule-based suggestions")
        elif self.api_key:
            print("Gemini API key detected")
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)
        else:
            print("No Gemini API key found - will use rule-based suggestions")
    
    @track_ai_performance
    def get_art_suggestions(self, drawing_type, emotion, previous_suggestions=None):
        """Get art suggestions based on the drawing and emotion"""
        if not self.api_available:
            # Fallback to rule-based suggestions if no API available
            return self.rule_based_suggestions(drawing_type, emotion)
        
        try:
            # Create more specific drawing context based on type
            drawing_descriptions = {
                "circle": "a circular shape",
                "square": "a square shape",
                "rectangle": "a rectangular shape",
                "triangle": "a triangular shape",
                "face": "a face with facial features",
                "person": "a person or stick figure",
                "tree": "a tree",
                "house": "a house or building",
                "sun": "a sun",
                "flower": "a flower",
                "heart": "a heart shape",
                "star": "a star shape",
                "abstract": "an abstract drawing",
                "landscape": "a landscape scene",
                "multiple_objects": "multiple shapes or objects",
                "dense_pattern": "an intricate pattern with many details",
            }
            
            drawing_context = drawing_descriptions.get(drawing_type, f"a {drawing_type}")
            emotion_context = f"The user is feeling {emotion}"
            
            # Include previous suggestions if available
            history_context = ""
            if previous_suggestions:
                history_context = f"\nPreviously, I suggested: {previous_suggestions}"
            
            # Craft the prompt for Gemini with clearer instructions
            prompt = f"""
You are an art therapy assistant. The user has just drawn {drawing_context}. {emotion_context}.{history_context}

Provide a brief, encouraging therapeutic suggestion (2-3 sentences) that:
1. Acknowledges what they drew specifically (e.g., "I see you've drawn {drawing_context}!")
2. Suggests ONE specific drawing tool (pencil, brush, spray, or eraser) and technique
3. Recommends TWO colors that complement their emotional state
4. Offers a specific element they could add to enhance their drawing

Be specific about the drawing type. Keep it warm, supportive, and actionable.
"""
            
            # Configure the model
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 150,
            }
            
            # Try gemini-2.0-flash first, fall back to gemini-pro if not available
            model_names = ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-pro"]
            
            for model_name in model_names:
                try:
                    # Create the model
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        generation_config=generation_config
                    )
                    
                    # Generate the response
                    response = model.generate_content(prompt)
                    
                    if response and hasattr(response, 'text'):
                        suggestion = response.text.strip()
                        # Include the text length in the return for better metrics tracking
                        return {
                            "suggestion": suggestion, 
                            "source": f"gemini ({model_name})", 
                            "text_length": len(suggestion)
                        }
                    else:
                        print(f"Empty response from Gemini API with model {model_name}")
                        continue
                        
                except Exception as model_error:
                    print(f"Error with model {model_name}: {str(model_error)}")
                    if model_name == model_names[-1]:  # Last model in list
                        # All models failed, use rule-based
                        return self.rule_based_suggestions(drawing_type, emotion)
                    continue  # Try next model
                
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return self.rule_based_suggestions(drawing_type, emotion)
    
    @track_ai_performance
    def get_detailed_art_suggestions(self, drawing_context, emotion, previous_suggestions=None):
        """Get detailed art suggestions based on more complete drawing analysis"""
        if not self.api_available:
            # Fallback to rule-based suggestions if no API available
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
            
            # Create specific drawing descriptions
            drawing_descriptions = {
                "circle": "a circular shape",
                "square": "a square",
                "rectangle": "a rectangle",
                "triangle": "a triangle",
                "face": "a face with facial features",
                "sun": "a sun",
                "tree": "a tree",
                "flower": "a flower",
                "heart": "a heart",
                "star": "a star",
                "landscape": "a landscape scene",
                "multiple_objects": "multiple shapes and objects",
                "dense_pattern": "an intricate, detailed pattern",
            }
            
            specific_drawing = drawing_descriptions.get(drawing_type, f"a {drawing_type}")
            
            drawing_details = f"The user has drawn {specific_drawing}"
            
            # Add color information if available
            if colors:
                drawing_details += f", using {', '.join(colors[:2])}"
            
            # Add composition details
            if density_map:
                if density_map[4] > sum(density_map) / 9 * 2:
                    drawing_details += ", centered on the canvas"
                elif density_map[0] + density_map[1] + density_map[3] > sum(density_map) / 2:
                    drawing_details += ", positioned in the upper-left area"
                elif density_map[2] + density_map[5] + density_map[8] > sum(density_map) / 2:
                    drawing_details += ", positioned on the right side"
            
            # Add detail level
            pixel_density = details.get("pixel_density", 0.0)
            if pixel_density < 0.05:
                drawing_details += ". The drawing is minimal with simple lines."
            elif pixel_density > 0.5:
                drawing_details += ". The drawing has rich detail and shading."
            else:
                drawing_details += ". The drawing has moderate detail."
                
            # Emotional context
            emotion_context = f"The user is currently feeling {emotion}."
            
            # Previous suggestions context
            history_context = ""
            if previous_suggestions:
                history_context = f"Previously, I suggested: {previous_suggestions}"
            
            # Craft the prompt for Gemini with detailed context
            prompt = f"""
You are an art therapy assistant. {drawing_details} The user is feeling {emotion}.
{history_context if previous_suggestions else ""}

Provide a warm, specific therapeutic suggestion (3-4 sentences) that:
1. Acknowledges their {specific_drawing} specifically
2. Suggests ONE specific drawing technique (like shading, hatching, blending, outlining, or adding texture)
3. Recommends ONE drawing tool (pencil, brush, spray, or eraser) for the technique
4. Suggests a specific element to add (like shadows, highlights, background elements, or details)
5. Recommends 1-2 colors that would work well with their drawing and emotional state

Be specific and encouraging. Don't use generic phrases like "what looks to be" - be confident about what they've drawn!
"""
            
            # Configure the model
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 200,
            }
            
            # Try gemini models in order of preference
            model_names = ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-pro"]
            
            for model_name in model_names:
                try:
                    # Create the model
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        generation_config=generation_config
                    )
                    
                    # Generate the response
                    response = model.generate_content(prompt)
                    
                    if response and hasattr(response, 'text'):
                        suggestion = response.text.strip()
                        # Include the text length in the return for better metrics tracking
                        return {
                            "suggestion": suggestion, 
                            "source": f"gemini ({model_name})", 
                            "text_length": len(suggestion)
                        }
                    else:
                        print(f"Empty response from Gemini API with model {model_name}")
                        continue
                        
                except Exception as model_error:
                    print(f"Error with model {model_name}: {str(model_error)}")
                    if model_name == model_names[-1]:  # Last model in list
                        # All models failed, use rule-based
                        return self.rule_based_suggestions(drawing_context["type"], emotion)
                    continue  # Try next model
                
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
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