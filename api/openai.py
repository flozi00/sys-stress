import requests # Still used for potential initial checks or sync fallback if needed
import time
import os
import tiktoken # Ensure you have tiktoken installed: pip install tiktoken
import datasets # Ensure you have datasets installed: pip install datasets
import random
import asyncio # For concurrency
import aiohttp # For async HTTP requests
from collections import deque # Efficiently manage prompts
import json # To parse streaming chunks and calculate payload size
import statistics # Added for median calculation

# --- Configuration ---
# Replace with your actual API endpoint URL
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
# Replace with your API key if required, otherwise set to None or ""
API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_IF_NEEDED")
# Model name to use for the benchmark
MODEL_NAME = "your-model-name" # e.g., "gpt-3.5-turbo", "mistralai/Mistral-7B-Instruct-v0.1"
# Dataset configuration
DATASET_NAME = "fka/awesome-chatgpt-prompts"
DATASET_SPLIT = "train"
PROMPT_COLUMN = "prompt"
# Number of TOTAL requests to send for benchmarking
TOTAL_REQUESTS = 200 # Adjust as needed
# Number of CONCURRENT requests (simulated users) to send at a time
# Setting CONCURRENT_USERS = TOTAL_REQUESTS means all requests are sent in one initial batch
CONCURRENT_USERS = TOTAL_REQUESTS # Adjust based on desired load simulation
# Maximum number of tokens to generate per request (adjust as needed)
MAX_TOKENS = 1024
# Optional: Temperature setting for generation
TEMPERATURE = 0.7
# Request timeout in seconds (overall request)
REQUEST_TIMEOUT = 90
# Timeout for reading data chunks during streaming (important!)
# STREAM_READ_TIMEOUT is not directly used by aiohttp's iter_any like this,
# REQUEST_TIMEOUT covers the whole operation. Keep for clarity if adapting later.
STREAM_READ_TIMEOUT = 60
# Optional delay between batches of concurrent requests (in seconds)
# Less relevant if CONCURRENT_USERS = TOTAL_REQUESTS
DELAY_BETWEEN_BATCHES = 0.0001 # Increase if hitting rate limits

# Encoding for token counting
try:
    ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    print("Warning: Could not get 'cl100k_base' encoding. Falling back to 'p50k_base'.")
    ENCODING = tiktoken.get_encoding("p50k_base")

# --- Helper Functions ---
def count_tokens(text):
    """Counts the number of tokens in a given text using the specified encoding."""
    if not text:
        return 0
    return len(ENCODING.encode(text))

def count_input_tokens_from_payload(payload):
    """Counts the input tokens based on the messages in the payload."""
    tokens = 0
    if "messages" in payload and isinstance(payload["messages"], list):
        for message in payload["messages"]:
            if isinstance(message, dict) and "content" in message and message["content"]:
                tokens += count_tokens(message["content"])
                tokens += 4 # Approximation for role, separators etc.
    tokens += 3 # Approximation for overall structure
    return tokens


# --- Load Dataset ---
def load_prompts(dataset_name, split, column, num_prompts):
    """Loads prompts from the specified Hugging Face dataset."""
    try:
        print(f"Loading dataset '{dataset_name}' split '{split}'...")
        dataset = datasets.load_dataset(dataset_name, split=split, streaming=True)
        print("Dataset loaded successfully (streaming).")

        prompts = []
        iterator = iter(dataset)
        count = 0
        # Ensure we don't try to load more prompts than requested if dataset is smaller
        print(f"Attempting to load {num_prompts} prompts...")
        while count < num_prompts:
            try:
                item = next(iterator)
                if item and column in item and item[column]:
                    prompts.append(item[column])
                    count += 1
                else:
                    print(f"Warning: Skipping invalid item or missing column '{column}'. Item: {item}")
            except StopIteration:
                print(f"Warning: Dataset exhausted after loading {count} prompts (requested {num_prompts}).")
                break
            except Exception as e:
                 print(f"Warning: Error reading item from streaming dataset: {e}")
                 continue # Skip problematic item

        if not prompts:
             print(f"Error: No valid prompts could be loaded from column '{column}'.")
             return None
        elif len(prompts) < num_prompts:
             print(f"Warning: Loaded only {len(prompts)} prompts, less than the requested {num_prompts}.")


        print(f"Successfully loaded {len(prompts)} prompts.")
        random.shuffle(prompts)
        return prompts

    except Exception as e:
        print(f"Error loading or processing dataset '{dataset_name}': {e}")
        print("Please ensure the 'datasets' and 'aiohttp' libraries are installed")
        print("and the dataset name/split are correct.")
        return None


# --- Asynchronous Request Function (Streaming Enabled) ---
async def send_request(session, url, headers, payload, request_num, total_requests):
    """Sends a single asynchronous streaming request and returns detailed results."""
    start_time = time.monotonic()
    first_token_time = None
    time_to_first_token = None
    full_response_content = ""
    input_tokens = count_input_tokens_from_payload(payload)
    output_tokens = 0
    status_code = None
    error_message = None
    response_received = False # Flag to track if any response part was received

    # Ensure stream is set to True for this function
    payload["stream"] = True

    print(f"Sending request {request_num}/{total_requests}... (Input Tokens: {input_tokens}, Streaming)")
    try:
        # Use a timeout object for more control if needed, but simple timeout is often sufficient
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with session.post(url, headers=headers, json=payload, timeout=timeout) as response:
            status_code = response.status
            response_received = True # Mark that we got a response header

            if response.status >= 200 and response.status < 300:
                # Process the stream
                async for line in response.content.iter_any(): # Use iter_any for bytes
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        # Handle potential multiple data chunks in one line
                        for segment in decoded_line.split('\n'):
                            segment = segment.strip()
                            if segment.startswith("data:"):
                                data_content = segment[len("data:"):].strip()
                                if data_content == "[DONE]":
                                    # print(f"Request {request_num}: Received [DONE] marker.")
                                    break # Exit inner loop for segments
                                try:
                                    chunk = json.loads(data_content)
                                    if chunk.get("choices"):
                                        delta = chunk["choices"][0].get("delta", {})
                                        content_piece = delta.get("content")
                                        if content_piece:
                                            # Record time of first actual content token
                                            if first_token_time is None:
                                                first_token_time = time.monotonic()
                                                time_to_first_token = first_token_time - start_time
                                                # print(f"Request {request_num}: First token received at {time_to_first_token:.4f}s")
                                            # Append content
                                            full_response_content += content_piece
                                except json.JSONDecodeError:
                                    print(f"Warning: Request {request_num} received non-JSON data chunk: {data_content}")
                                    continue
                                except Exception as e:
                                    print(f"Warning: Request {request_num} error processing chunk: {e} | Chunk: {data_content}")
                                    continue
                        # Check again if DONE marker was processed to break outer loop
                        if data_content == "[DONE]":
                            break

            else:
                # Handle non-2xx status codes even before streaming starts
                response_text = await response.text()
                error_message = f"HTTP Status {status_code}: {response_text}"
                print(f"Error in request {request_num}: {error_message}")

    except asyncio.TimeoutError:
        if not response_received:
             error_message = f"Request {request_num} connection timed out after {REQUEST_TIMEOUT}s (before response)"
        else:
             error_message = f"Request {request_num} streaming/total timed out after {REQUEST_TIMEOUT}s"
        print(error_message)
        status_code = 408 # Request Timeout status code
    except aiohttp.ClientError as e:
        error_message = f"ClientError during request {request_num}: {e}"
        print(error_message)
        status_code = getattr(e, 'status', None) or getattr(e, 'code', None) or 500
    except Exception as e:
        error_message = f"Unexpected error during request {request_num}: {type(e).__name__}: {e}"
        print(error_message)
        status_code = status_code or 500 # Keep original status if available, else 500

    end_time = time.monotonic()
    total_request_duration = end_time - start_time

    # Calculate output tokens based on the fully accumulated content
    if full_response_content:
        output_tokens = count_tokens(full_response_content)
    else:
        output_tokens = 0
        if status_code == 200 and error_message is None: # Check if successful but no content
            print(f"Warning: Request {request_num} completed successfully but no content was streamed.")
            # error_message = "Successful response but no content received" # Uncomment if this is an error


    success = error_message is None and status_code == 200 and output_tokens > 0 # Consider success only if content received

    # print(f"Request {request_num}: Completed. Status={status_code}, Total Time={total_request_duration:.4f}s, TTFT={time_to_first_token if time_to_first_token else 'N/A'}s, Input Tokens={input_tokens}, Output Tokens={output_tokens}, Success={success}")

    return {
        "total_duration": total_request_duration,
        "time_to_first_token": time_to_first_token, # Can be None if no token received
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "success": success,
        "status_code": status_code,
        "error": error_message
    }


# --- Main Benchmarking Function (Async) ---
async def benchmark_api_concurrent(prompts):
    """Sends requests concurrently to the API endpoint and calculates metrics."""
    if not prompts:
        print("No prompts provided for benchmarking. Exiting.")
        return

    # Adjust total_requests_to_send based on the number of prompts actually loaded
    total_requests_to_send = len(prompts)
    if total_requests_to_send == 0:
        print("No prompts loaded, cannot run benchmark.")
        return
    print(f"Starting benchmark with {total_requests_to_send} requests.")

    results = []
    prompts_queue = deque(prompts)

    headers = {
        "Content-Type": "application/json",
    }
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    # Use a single session for connection pooling
    # Increase connector limit if needed for high concurrency
    connector = aiohttp.TCPConnector(limit=CONCURRENT_USERS + 10) # Allow slightly more connections than users
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        start_benchmark_time = time.monotonic()
        request_counter = 0
        tasks = [] # Keep track of all tasks

        # Launch all tasks up to the concurrency limit or total requests
        for i in range(total_requests_to_send):
             if not prompts_queue:
                 print("Warning: Ran out of prompts unexpectedly.")
                 break
             request_counter += 1
             prompt_text = prompts_queue.popleft()
             payload = {
                 "model": MODEL_NAME,
                 "messages": [{"role": "user", "content": prompt_text}],
                 "max_tokens": MAX_TOKENS,
                 "temperature": TEMPERATURE,
                 # "stream" will be set to True inside send_request
             }
             task = asyncio.create_task(
                 send_request(session, API_URL, headers, payload, request_counter, total_requests_to_send)
             )
             tasks.append(task)

             # Optional: Add delay if not sending all concurrently at once
             # This part is less relevant if CONCURRENT_USERS = TOTAL_REQUESTS
             if i > 0 and i % CONCURRENT_USERS == 0 and DELAY_BETWEEN_BATCHES > 0:
                 print(f"Delaying for {DELAY_BETWEEN_BATCHES}s after launching batch...")
                 await asyncio.sleep(DELAY_BETWEEN_BATCHES)


        print(f"Launched all {len(tasks)} request tasks. Waiting for completion...")
        # Wait for all launched tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_benchmark_time = time.monotonic()
        total_benchmark_duration = end_benchmark_time - start_benchmark_time
        print(f"All {len(results)} requests finished.")


    # --- Process Results ---
    successful_requests = 0
    failed_requests = 0
    total_input_tokens = 0  # Sum of input tokens for all requests attempted
    total_output_tokens = 0 # Sum of output tokens for successful requests
    total_request_duration_all = 0.0 # Sum of total durations for all requests
    total_ttft = 0.0 # Sum of TTFT for successful requests where TTFT is available
    ttft_count = 0 # Count of requests where TTFT was measured
    latencies_ttft = [] # List of TTFT values for successful requests
    latencies_total = [] # List of total duration values for all requests

    # *** NEW: Lists to store per-request TPS values ***
    per_request_input_tps_ttft = []
    per_request_output_tps_post_ttft = []

    error_summary = {}

    print("Processing results...")
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            failed_requests += 1
            print(f"Result {i+1}: System Error processing task: {res}")
            error_msg = str(res)
            error_summary[error_msg] = error_summary.get(error_msg, 0) + 1
        elif isinstance(res, dict):
            latencies_total.append(res['total_duration'])
            total_request_duration_all += res['total_duration']
            # Count input tokens regardless of success, as they were sent/processed by the script
            current_input_tokens = res.get('input_tokens', 0)
            total_input_tokens += current_input_tokens

            if res['success']:
                successful_requests += 1
                current_output_tokens = res.get('output_tokens', 0)
                total_output_tokens += current_output_tokens

                if res.get('time_to_first_token') is not None:
                    ttft = res['time_to_first_token']
                    total_ttft += ttft
                    latencies_ttft.append(ttft)
                    ttft_count += 1

                    # *** NEW: Calculate per-request TPS ***
                    if ttft > 0 and current_input_tokens > 0:
                        req_input_tps = current_input_tokens / ttft
                        per_request_input_tps_ttft.append(req_input_tps)

                    generation_time = res['total_duration'] - ttft
                    if generation_time > 0 and current_output_tokens > 0:
                        req_output_tps = current_output_tokens / generation_time
                        per_request_output_tps_post_ttft.append(req_output_tps)

            else:
                failed_requests += 1
                error_msg = res.get('error', 'Unknown Error')
                status = res.get('status_code', 'N/A')
                key = f"Status {status}: {error_msg}"
                error_summary[key] = error_summary.get(key, 0) + 1
        else:
             print(f"Warning: Result {i+1}: Unexpected result type received: {type(res)}")
             failed_requests += 1

    # --- Calculate and Print Metrics ---
    print("\n--- Concurrent Streaming Benchmark Results ---")
    print(f"Concurrency Level:       {CONCURRENT_USERS} users")
    print(f"Target Total Requests:   {TOTAL_REQUESTS}") # Use original target
    print(f"Actual Requests Sent:    {len(tasks)}")    # How many tasks were created
    print(f"Actual Results Received: {len(results)}")  # How many results processed
    print(f"Successful Requests:     {successful_requests} (requests with received content)")
    print(f"Failed Requests:         {failed_requests}")
    print(f"Requests with TTFT:      {ttft_count}") # How many requests successfully reported TTFT

    if error_summary:
        print("\nError Summary:")
        for error, count in sorted(error_summary.items()): # Sort for consistent output
            print(f"- {error}: {count} times")

    # Initialize metrics to avoid NameError if no successful requests
    average_total_latency = 0.0
    average_ttft = 0.0
    requests_per_second = 0.0
    median_input_tps_ttft = "N/A" # Use string N/A if cannot be calculated
    median_output_tps_post_ttft = "N/A"
    input_tps_per_user = 0.0
    output_tps_per_user = 0.0 # Overall output TPS / user

    if len(results) > 0 and total_benchmark_duration > 0:
        requests_per_second = len(results) / total_benchmark_duration # RPS based on all attempts
        average_total_latency = total_request_duration_all / len(results) # Avg total latency over all attempts

    if ttft_count > 0: # Use ttft_count for average TTFT
        average_ttft = total_ttft / ttft_count

    # *** NEW: Calculate Median TPS ***
    if per_request_input_tps_ttft:
        median_input_tps_ttft = statistics.median(per_request_input_tps_ttft)
    if per_request_output_tps_post_ttft:
        median_output_tps_post_ttft = statistics.median(per_request_output_tps_post_ttft)


    # Calculate overall TPS and per-user TPS (based on total duration, similar to non-streaming)
    overall_input_tps = 0.0
    overall_output_tps = 0.0
    if total_benchmark_duration > 0:
        # Base overall input TPS on all input tokens processed over the total benchmark time
        overall_input_tps = total_input_tokens / total_benchmark_duration
         # Base overall output TPS on successful output tokens over the total benchmark time
        overall_output_tps = total_output_tokens / total_benchmark_duration

    if CONCURRENT_USERS > 0:
        input_tps_per_user = overall_input_tps / CONCURRENT_USERS
        output_tps_per_user = overall_output_tps / CONCURRENT_USERS


    print("\nPerformance Metrics:")
    print(f"Total Benchmark Duration:  {total_benchmark_duration:.4f} seconds")
    print(f"Requests Per Second (RPS): {requests_per_second:.4f} (based on all attempts)")
    print(f"Average Total Latency:   {average_total_latency:.4f} seconds (per request, all attempts)")
    if ttft_count > 0:
        print(f"Average Time To First Token (TTFT): {average_ttft:.4f} seconds (for {ttft_count} successful streams)")
    else:
        print("Average Time To First Token (TTFT): N/A (no successful streams measured)")
    print("-" * 30)
    print(f"Total Input Tokens Sent:   {total_input_tokens}")
    print(f"Total Output Tokens Gen.:  {total_output_tokens} (from successful streams)")
    print("-" * 30)
    # Overall TPS (useful for comparing total throughput)
    print(f"Overall Input Tokens/Sec:  {overall_input_tps:.4f}")
    print(f"Overall Output Tokens/Sec: {overall_output_tps:.4f}")
    print("-" * 30)
    # *** UPDATED: Streaming Specific TPS using MEDIAN ***
    if isinstance(median_input_tps_ttft, float):
        print(f"Median Input Tokens/Sec (until First): {median_input_tps_ttft:.4f} (per request)")
    else:
        print(f"Median Input Tokens/Sec (until First): {median_input_tps_ttft}")

    if isinstance(median_output_tps_post_ttft, float):
        print(f"Median Output Tokens/Sec (after First): {median_output_tps_post_ttft:.4f} (per request)")
    else:
        print(f"Median Output Tokens/Sec (after First): {median_output_tps_post_ttft}")
    print("-" * 30)
    # Per User TPS (based on overall TPS)
    print(f"Avg Input Tokens/Sec/User:  {input_tps_per_user:.4f}")
    print(f"Avg Output Tokens/Sec/User: {output_tps_per_user:.4f}")


    print("------------------------------------\n")


# --- Run the benchmark ---
if __name__ == "__main__":
    # Basic validation for configuration
    if "YOUR_API_ENDPOINT_URL" in API_URL or API_URL == "http://127.0.0.1:8000/v1/chat/completions": # Added check for default URL
        print("Warning: API_URL is set to the default or a placeholder.")
        print("Please replace it with your actual API endpoint in the script.")
        # Optionally exit if the URL is critical and not set
        # import sys
        # sys.exit("Exiting: API URL not configured.")

    if TOTAL_REQUESTS <= 0 or CONCURRENT_USERS <= 0:
         print("Error: TOTAL_REQUESTS and CONCURRENT_USERS must be positive integers.")
    else:
        # Load prompts, ensuring we use the actual number loaded if dataset is smaller
        prompts_to_use = load_prompts(DATASET_NAME, DATASET_SPLIT, PROMPT_COLUMN, TOTAL_REQUESTS)

        # Run benchmark only if prompts were loaded successfully
        if prompts_to_use and len(prompts_to_use) > 0:
             # Adjust TOTAL_REQUESTS if fewer prompts were loaded than initially requested
             if len(prompts_to_use) < TOTAL_REQUESTS:
                 print(f"Adjusting benchmark to run with {len(prompts_to_use)} loaded prompts.")
                 TOTAL_REQUESTS = len(prompts_to_use) # Update global or pass to function if preferred

             # Run the async benchmark function
             asyncio.run(benchmark_api_concurrent(prompts_to_use))
        elif prompts_to_use is None:
             print("Benchmark aborted due to errors loading prompts.")
        else: # prompts_to_use is an empty list
             print("Benchmark aborted as no prompts were loaded.")
