
def prompt_with_context(
        previous_analysis: str,
        prompt_single_image: str
) -> str:

    prompt_with_previous_context = f"""
    Previous analysis: {previous_analysis}

    Now analyze the next frame in the sequence:
    {prompt_single_image}
    """
    return prompt_with_previous_context


HAZARD_DETECTION_PROMPT_SINGLE_FRAME = """Analyze this frame from a car's perspective - the camera is mounted on the vehicle, 
giving a first-person view of the surroundings (you cannot see the car you're mounted on).

Theft examples relevant to this view:
- People approaching with suspicious intent
- People reaching into the car through the window, grabbing items

Robbery examples relevant to this view:
- Armed individuals appearing in your field of view
- Masked individuals approaching your position
- Threatening behavior directed towards your position
- Groups showing aggressive behavior in your immediate surroundings
- Individuals fighting

Output format required as JSON:
{
    "hazard_detected": true/false,
    "hazard_type": "none" or ["theft" or "robbery"],
    "reasoning": "One sentence explaining why"
}

Important:
- Only report clear evidence of theft or robbery threats to your position
- Ignore activity around other parked cars
- Normal pedestrian traffic passing by is not suspicious
- Regular parking lot activities are not hazards
- If in doubt or unclear, report no hazard
- Keep reasoning to one clear sentence
- Focus only on threats approaching your position/field of view

Be conservative in assessment - if the situation shows normal activity in your surroundings, report no hazard."""


HAZARD_DETECTION_MULTIPLE_FRAMES_SINGLE_BATCH_V1 = """Analyze this mosaic plot, which contains a grid plot of frames from 
a car's camera mounted on the vehicle.  Read the frames sequentially from top-left to bottom-right, and analyze them as 
a continuous sequence of events.

Theft examples relevant to this mosaic plot:
- People approaching with suspicious intent
- People reaching into the car through the window, grabbing items

Robbery examples relevant to this mosaic plot:
- Armed individuals appearing in your field of view
- Masked individuals approaching your position
- Threatening behavior directed towards your position
- Groups showing aggressive behavior in your immediate surroundings
- Individuals fighting

For the overall mosaic plot, provide a **separate analyses** in **JSON format** as follows:

**Analysis:**
{
    "hazard_detected": true/false,
    "hazard_type": "none" or "theft" or "robbery",
    "reasoning": "One sentence explaining why"
}

Important:

- Only report clear evidence of theft or robbery threats to your position.
- Ignore activity around other parked cars.
- Normal pedestrian traffic passing by is not suspicious.
- Regular parking lot activities are not hazards.
- If in doubt, unclear, or if no hazards are detected, set `hazard_detected` to false.
- Focus only on threats approaching your position/field of view.
- Disregard any visual noise or irrelevant details that do not contribute to a clear hazard identification.

Be conservative in assessment - if the situation shows normal activity in your surroundings, report no hazard.

Ensure the output strictly adheres to the specified JSON format without additional commentary or deviation.
"""


HAZARD_DETECTION_MULTIPLE_FRAMES_SINGLE_BATCH = """Analyze this mosaic plot showing frames from a car's camera mounted on the vehicle. Read the frames sequentially from top-left to bottom-right as a continuous sequence of events.

Theft examples:
- People approaching with suspicious intent
- People reaching into the car through windows, grabbing items

Robbery examples:
- Armed individuals in field of view
- Masked individuals approaching
- Threatening behavior directed at vehicle
- Groups showing aggressive behavior nearby
- Individuals fighting

Provide analysis in JSON format:

{
    "hazard_detected": true/false,
    "hazard_type": "none" or "theft" or "robbery",
    "reasoning": "One sentence explaining why"
}

Guidelines:
- Only report clear evidence of theft or robbery threats
- Ignore activity around other parked cars
- Normal pedestrian traffic is not suspicious
- Regular parking lot activities are not hazards
- If unclear, set hazard_detected to false
- Focus only on direct threats to the vehicle
- Disregard irrelevant visual details

Output must strictly follow the JSON format without additional commentary."""


HAZARD_DETECTION_MULTIPLE_FRAMES_SINGLE_BATCH_CoT ="""Analyze this mosaic plot showing frames from a car's camera mounted on the vehicle. Read the frames sequentially from top-left to bottom-right as a continuous sequence of events.

Step 1 - Describe the key activities visible in each frame:
{
    "frame_analysis": [
        {"frame": 1, "activity": "Description of activity"},
        {"frame": 2, "activity": "Description of activity"}
    ]
}

Step 2 - Identify potential concerning behaviors:
{
    "concerns": [
        {
            "frame_numbers": [],
            "behavior": "Description of concerning behavior",
            "matches_category": "theft/robbery/none"
        }
    ]
}

Step 3 - Final hazard assessment:
{
    "hazard_detected": false,
    "hazard_type": "none",
    "reasoning": "One sentence synthesizing the analysis from steps 1 and 2",
    "confidence": "high"
}

Reference categories:
Theft indicators:
- People approaching with suspicious intent
- People reaching into car through windows, grabbing items

Robbery indicators:
- Armed individuals in field of view
- Masked individuals approaching
- Threatening behavior directed at vehicle
- Groups showing aggressive behavior nearby
- Individuals fighting

Guidelines:
- Describe each frame objectively before making interpretations
- Track movements and behaviors across sequential frames
- Only escalate to hazard if behavior clearly matches theft/robbery indicators
- Ignore normal activities (pedestrians, parking)
- If in doubt, indicate lower confidence and no hazard
- Focus only on direct threats to the vehicle"""