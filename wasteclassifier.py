from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from inference_sdk import InferenceHTTPClient
import io
from collections import defaultdict

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

waste_classification = {
    "cardboard boxes": "Biodegradable",
    "glass bottles": "NonBiodegradable but Recyclable",
    "glass shards": "NonBiodegradable but Recyclable",
    "jug": "NonBiodegradable but Recyclable",
    "metal can": "NonBiodegradable but Recyclable",
    "other plastic": "NonBiodegradable and Non-Recyclable",
    "paper": "Biodegradable",
    "paper bag": "Biodegradable",
    "paper carton": "Biodegradable",
    "paper cup": "Biodegradable",
    "plastic bottle": "NonBiodegradable but Recyclable",
    "plastic cap": "NonBiodegradable but Recyclable",
    "plastic cup": "NonBiodegradable and Non-Recyclable",
    "plastic utensils": "NonBiodegradable and Non-Recyclable",
    "plastic bag": "NonBiodegradable and Non-Recyclable",
    "styrofoam": "NonBiodegradable and Non-Recyclable",
    "unlabeled": "NonBiodegradable and Non-Recyclable",
    "food waste": "Biodegradable"
}

# Solutions based on waste classification categories
waste_solutions = {
    "Biodegradable": {
        "disposal": "Compost bin or organic waste collection",
        "benefits": "Can decompose naturally and return nutrients to soil",
        "tips": "Consider home composting or municipal green waste programs",
        "impact": "Reduces landfill waste and greenhouse gas emissions",
        "alternatives": "These materials are generally eco-friendly, but consider using reusable options when possible",
        "advanced_options": "Bokashi composting for food waste or vermicomposting for apartment living"
    },
    "NonBiodegradable but Recyclable": {
        "disposal": "Recycling bin (clean and dry)",
        "benefits": "Can be processed into new products, saving resources",
        "tips": "Rinse containers, remove labels when possible, and check local recycling guidelines",
        "impact": "Reduces raw material extraction and energy consumption",
        "alternatives": "Consider reusable alternatives like glass containers or stainless steel bottles",
        "additional_resources": "TerraCycle programs for hard-to-recycle items or local recycling drop-off centers"
    },
    "NonBiodegradable and Non-Recyclable": {
        "disposal": "General waste bin (landfill)",
        "alternatives": "Look for reusable alternatives or products with better recyclability",
        "reduction": "Try to minimize use of these materials when possible",
        "impact": "These items persist in the environment for many years",
        "advocacy": "Support policies that require producers to create more sustainable packaging",
        "upcycling": "Before disposal, consider creative reuse options or donating to art programs"
    }
}

# Additional item-specific disposal instructions
specific_item_solutions = {
    "plastic bottle": "Remove cap (recycle separately), rinse, and crush to save space. Consider reusing for DIY projects or as a water bottle.",
    "plastic cap": "Check local recycling guidelines; some facilities require caps to be separated, while others accept them attached to bottles.",
    "paper cup": "Remove plastic lining if possible before composting. Consider switching to a reusable travel mug.",
    "paper bag": "Reuse for storage, gift wrapping, or composting. Fold and keep for future shopping trips.",
    "paper carton": "Rinse, flatten, and recycle. Some communities require removal of plastic spouts.",
    "cardboard boxes": "Remove tape and flatten before recycling. Can be repurposed for storage or composting.",
    "styrofoam": "Some communities have special styrofoam recycling programs. Use as packaging material when shipping items.",
    "glass bottles": "Rinse thoroughly and recycle; colored and clear glass may need to be separated. Can be upcycled into vases or containers.",
    "glass shards": "Carefully wrap in paper before disposal to prevent injuries to waste handlers. Some glass recycling programs accept broken glass.",
    "jug": "Clean thoroughly, remove labels when possible, and check if your recycling program accepts the specific type of plastic.",
    "metal can": "Rinse clean, remove paper labels if required by local guidelines. Crush to save space in recycling bins.",
    "plastic bag": "Many grocery stores offer plastic bag recycling drop-offs. Reuse for trash liners or shopping.",
    "plastic cup": "Reduce usage by switching to reusable cups. Some specialized recycling programs may accept them.",
    "plastic utensils": "Consider switching to compostable alternatives or carrying reusable cutlery. Some specialized recycling programs exist.",
    "other plastic": "Check for recycling symbol and number to determine recyclability in your area. Minimize purchase of mixed-material plastics.",
    "food waste": "Ideal for composting; separate into green waste bins if available. Consider worm composting for apartment living.",
    "paper": "Shred sensitive documents before recycling. Use as mulch in garden or compost.",
    "unlabeled": "Attempt to identify material type before disposal. If uncertain, check with local waste management authority for guidance."
}

@app.post("/waste_classification")
async def detect_waste(image: UploadFile = File(...)):
    image_bytes = await image.read()
    CLIENT = InferenceHTTPClient(
        api_url="https://classify.roboflow.com",
        api_key="umOiGbjhCrKijvMJwo1H"
    )
    img = Image.open(io.BytesIO(image_bytes))
    result = CLIENT.infer(img, model_id="waste-classification-rjo28/2")
    
    response_data = {}
    
    for prediction in result['predicted_classes']:
        item_lower = str(prediction).lower()
        category = waste_classification[item_lower]
        
        # Get local recycling information based on waste type
        local_resources = get_local_recycling_info(item_lower)
        
        item_response = {
            "classification": category,
            "general_solution": waste_solutions[category],
            "local_resources": local_resources
        }
        
        # Add specific solution for the item if available
        if item_lower in specific_item_solutions:
            item_response["specific_instructions"] = specific_item_solutions[item_lower]
        
        # Add DIY recycling or upcycling projects
        item_response["diy_projects"] = get_diy_projects(item_lower)
        
        response_data[prediction] = item_response
    
    return JSONResponse(content={"prediction": response_data})

# Function to provide local recycling information
def get_local_recycling_info(item_type):
    # This would ideally connect to a database or API with location-specific recycling information
    # Placeholder for demonstration purposes
    recycling_info = {
        "plastic bottle": [
            "Check Earth911.com for local recycling centers",
            "Many curbside recycling programs accept PET and HDPE plastic bottles"
        ],
        "paper": [
            "Most curbside recycling programs accept paper products",
            "Office supply stores often collect paper for recycling"
        ],
        "glass bottles": [
            "Glass recycling drop-off locations can be found at most recycling centers",
            "Some states have bottle deposit programs for glass containers"
        ],
        "metal can": [
            "Aluminum cans may qualify for cash redemption at recycling centers",
            "Steel cans are accepted in most curbside recycling programs"
        ]
    }
    
    # Return general recycling information for items not specifically listed
    if item_type not in recycling_info:
        return [
            "Search your local municipal waste management website for specific guidelines",
            "Contact your waste hauler for information on special waste handling",
            "Visit Earth911.com to find recycling options near you"
        ]
    
    return recycling_info[item_type]

# Function to suggest DIY or upcycling projects
def get_diy_projects(item_type):
    diy_projects = {
        "plastic bottle": [
            "Self-watering planter: Cut bottle in half, invert top into bottom, and fill with soil and plant",
            "Bird feeder: Cut openings and add perches for birds to access seeds",
            "Piggy bank: Decorate and cut a slot for coins"
        ],
        "cardboard boxes": [
            "Storage organizers: Cut and fold to create drawer dividers",
            "Children's playhouse or fort: Connect multiple boxes for a fun structure",
            "Garden seed starters: Line with paper and fill with soil"
        ],
        "glass bottles": [
            "Decorative vase: Clean, remove labels, and paint or decorate",
            "Oil or soap dispenser: Add a pour spout or pump mechanism",
            "Garden border: Partially bury bottles along garden edges"
        ],
        "paper": [
            "Handmade recycled paper: Tear into small pieces, soak, blend, and dry on a screen",
            "Fire starters: Roll tightly for use in fireplace or campfire",
            "Paper mache art projects: Mix with glue to create sculptures"
        ],
        "metal can": [
            "Pen/pencil holder: Clean, remove sharp edges, and decorate",
            "Garden planters: Add drainage holes and paint",
            "Luminaries: Create patterns of holes for light to shine through"
        ]
    }
    
    # Return general upcycling suggestions for items not specifically listed
    if item_type not in diy_projects:
        return [
            "Search online platforms like Pinterest for upcycling ideas",
            "Consider donating usable items to schools for art projects",
            "Look for local creative reuse centers that accept donations of materials"
        ]
    
    return diy_projects[item_type]