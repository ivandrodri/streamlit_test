
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


HAZARD_DETECTION_PROMPT = """Analyze this frame from a car's perspective - the camera is mounted on the vehicle, 
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


HAZARD_DETECTION_GRID_PLOT_NUMBERED_PROMPT_v2 = """Analyze this mosaic plot, which contains a 5x5 grid of frames from a car's 
camera mounted on the vehicle. The frames are numbered from 0 (top-left) to 24 (bottom-right).

Theft examples relevant to this view:
- People approaching with suspicious intent
- People reaching into the car through the window, grabbing items

Robbery examples relevant to this view:
- Armed individuals appearing in your field of view
- Masked individuals approaching your position
- Threatening behavior directed towards your position
- Groups showing aggressive behavior in your immediate surroundings
- Individuals fighting

For the overall mosaic plot, provide **four separate analyses** in **JSON format** as follows:

**Analysis 1:**
{
    "hazard_detected": true/false,
    "frame_indices_where_hazard_is_detected": [frame_number_1, frame_number_2, ...],
    "hazard_type": "none" or "theft" or "robbery",
}

**Analysis 2:**
{
    "hazard_detected": true/false,
    "frame_indices_where_hazard_is_detected": [frame_number_1, frame_number_2, ...],
    "hazard_type": "none" or "theft" or "robbery",
}

**Analysis 3:**
{
    "hazard_detected": true/false,
    "frame_indices_where_hazard_is_detected": [frame_number_1, frame_number_2, ...],
    "hazard_type": "none" or "theft" or "robbery",
}

**Analysis 4:**
{
    "hazard_detected": true/false,
    "frame_indices_where_hazard_is_detected": [frame_number_1, frame_number_2, ...],
    "hazard_type": "none" or "theft" or "robbery",
}

Important:
- Make sure to assess each frame individually. If a frame contains a hazard, include its index in the `frame_indices_where_hazard_is_detected` array.
- Only report clear evidence of theft or robbery threats to your position.
- Ignore activity around other parked cars.
- Normal pedestrian traffic passing by is not suspicious.
- Regular parking lot activities are not hazards.
- If in doubt, unclear, or if no hazards are detected, set `hazard_detected` to false.
- Focus only on threats approaching your position/field of view.

Be conservative in assessment - if the situation shows normal activity in your surroundings, report no hazard.
"""



HAZARD_DETECTION_GRID_PLOT_NUMBERED_PROMPT = """Analyze this mosaic plot, which contains a 5x5 grid of frames from a car's 
camera mounted on the vehicle. The frames are numbered from 0 (top-left) to 24 (bottom-right).

Theft examples relevant to this view:
- People approaching with suspicious intent
- People reaching into the car through the window, grabbing items

Robbery examples relevant to this view:
- Armed individuals appearing in your field of view
- Masked individuals approaching your position
- Threatening behavior directed towards your position
- Groups showing aggressive behavior in your immediate surroundings
- Individuals fighting

For the overall mosaic plot, provide the following analysis in **JSON format**:
{
    "hazard_detected": true/false,
    "frame_indices_where_hazard_is_detected": [frame_number_1, frame_number_2, ...],
    "hazard_type": "none" or "theft" or "robbery",
    "reasoning": "One sentence explaining the reason for hazard detection, focusing on frames with hazards."
}

Important:
- Make sure to assess each frame individually. If a frame contains a hazard, include its index in the `frame_indices_where_hazard_is_detected` array.
- Only report clear evidence of theft or robbery threats to your position.
- Ignore activity around other parked cars.
- Normal pedestrian traffic passing by is not suspicious.
- Regular parking lot activities are not hazards.
- If in doubt, unclear, or if no hazards are detected, set `hazard_detected` to false and provide an explanation in `reasoning`.
- Keep reasoning to one clear sentence.
- Focus only on threats approaching your position/field of view.

Be conservative in assessment - if the situation shows normal activity in your surroundings, report no hazard.
"""


HAZARD_DETECTION_GRID_PLOT_PROMPT = """Analyze the provided mosaic plot from a car's perspective. 
The camera is mounted on the vehicle, providing a first-person view of the surroundings 
(you cannot see the car you're mounted on).

The image is a mosaic plot of frames from the car's video camera, arranged in a grid. Read the frames 
sequentially from top-left to bottom-right, and analyze them as a continuous sequence of events.

Theft examples relevant to this view:
- People approaching with suspicious intent.
- People reaching into the car through the window, grabbing items.

Robbery examples relevant to this view:
- Armed individuals appearing in your field of view.
- Masked individuals approaching your position.
- Threatening behavior directed towards your position.
- Groups showing aggressive behavior in your immediate surroundings.
- Individuals fighting.

**Output format required as JSON:**

{
    "hazard_detected": true/false,
    "hazard_type": "none" or ["theft", "robbery"],
    "reasoning": "One sentence explaining why"
}

**Important:**
- Only report clear evidence of theft or robbery threats to your position.
- Ignore activity around other parked cars.
- Normal pedestrian traffic passing by is not suspicious.
- Regular parking lot activities are not hazards.
- If in doubt or unclear, report no hazard.
- Keep reasoning to one clear sentence.
- Focus only on threats approaching your position/field of view.

Be conservative in your assessment. If the situation shows normal activity in your surroundings, report no hazard.
"""


SIMPLE_PROMPT_SMALL_MODELS = """
Instructions: You are analyzing a mosaic plot made up of sequential video frames from a car’s camera. The camera is mounted on the car, providing a view of the surroundings. Your task is to determine if there is a potential hazard related to theft or robbery based on the visual information in these frames.

Steps:
1. Examine each frame in the mosaic plot from top-left to bottom-right.
2. Look for specific signs of theft or robbery, such as:
   - Individuals reaching into the car.
   - Suspicious behavior like loitering near the car with apparent intent.
   - Individuals with weapons or wearing masks approaching the car.
   - Aggressive or threatening actions directed towards the camera.

Output Format (JSON):
{{
    "hazard_detected": true/false,
    "hazard_type": "none" or ["theft", "robbery"],
    "reasoning": "A clear, concise sentence explaining why"
}}

Important Notes:
- Only report hazards that directly threaten the car or the camera's field of view.
- Ignore normal pedestrian activity and actions related to other cars.
- Be conservative in your assessment. Report no hazard if the situation is unclear or normal.
"""