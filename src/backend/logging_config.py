import logging
import os


LOG_FOLDER = 'logs'
LOG_FILE = 'usage_logs.log'


if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)
log_path = os.path.join(LOG_FOLDER, LOG_FILE)


logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def log_llm_usage(cost: float, input_tokens: int, output_tokens: int):
    logging.info(f'Cost: ${cost}, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}')


def compute_total_llm_costs() -> float:
    total_cost = 0.0
    with open(log_path, 'r') as log_file:
        for line in log_file:
            if 'Cost:' in line:
                # Extract the cost value from the log line
                cost_str = line.split('Cost: $')[1].split(',')[0]
                total_cost += float(cost_str)
    return total_cost
