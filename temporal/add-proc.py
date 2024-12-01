from temporalio import activity, workflow, exceptions
from temporalio.client import Client
from temporalio.worker import Worker
import os

import json
from typing import Dict
from datetime import timedelta

# Define activities
@activity.defn
async def read_excel() -> str:
    # Import pandas inside the activity to avoid sandbox conflicts
    import pandas as pd
    df = pd.read_excel('/Users/menachem.geva/code/python/AI/address_obfuscation_cases-taggings.xlsx')
    # handle missing values
    df['root_address2'] = df['root_address2'].fillna("").astype(str)
    output_dir = "add-proc-data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "stage1_raw.parquet")
    df.to_parquet(output_path)
    return output_path

@activity.defn
async def filter_data(input_path: str) -> str:
    # Import pandas inside the activity
    import pandas as pd
    df = pd.read_parquet(input_path)
    filtered_df = df[(df['obfuscation_level']==3) & 
                    (df['similar_formatted_address'].isna()) & 
                    (df['ekata_formatted_address'].isna())]
    output_path = os.path.join("add-proc-data", "stage2_filtered.parquet")
    filtered_df.to_parquet(output_path)
    return output_path

@activity.defn
async def add_llma_column(input_path: str) -> str:
    # Import pandas inside the activity
    import pandas as pd
    df = pd.read_parquet(input_path)
    
    def get_address_string(row):
        return ' '.join([
            str(row['root_address1']), 
            str(row['root_address2']),
            str(row['root_city']),
            str(row['root_province_code']),
            str(row['root_zip']),
            'US'
        ])
    
    def get_addr_elaborate(addr):
        import ollama
        prompt = (
            "You are an assistant for formatting postal addresses. "
            "Please reformat the following address into a complete, single-line postal format. "
            "Do not add explanations or disclaimers. "
            "make sure to add the number in street if you get it."
            "if you get 0 in words it probably means o."
            "if you have - inside a word you can probably omit it."
            'give me back json in this format: {"Standardize_address" : Standardize this address,"score_fraud": your score between 0.0 to 1.0, "fraud_reason": reason you susspect address}'
            "Standardize json: "
        )
        
        try:
            response = ollama.generate(
                model='llama3.2',
                prompt=prompt + addr
            )
            return response['response']
        except Exception as e:
            print(f"Error processing address {addr}: {str(e)}")
            return json.dumps({
                "Standardize_address": "",
                "score_fraud": 0.0,
                "fraud_reason": f"Error: {str(e)}"
            })
    
    def parse_llma_response(response: str) -> Dict:
        try:
            return json.loads(response)
        except:
            return {
                "Standardize_address": "",
                "score_fraud": 0.0,
                "fraud_reason": "Failed to parse response"
            }
    
    df['address_string'] = df.apply(get_address_string, axis=1)
    df['llma_data'] = df['address_string'].apply(get_addr_elaborate)
    df['llma_data'] = df['llma_data'].apply(parse_llma_response)
    
    # Extract individual fields from the llma_data dictionary
    df['standardized_address'] = df['llma_data'].apply(lambda x: x.get('Standardize_address', ''))
    df['fraud_score'] = df['llma_data'].apply(lambda x: x.get('score_fraud', 0.0))
    df['fraud_reason'] = df['llma_data'].apply(lambda x: x.get('fraud_reason', ''))
    
    output_path = os.path.join("add-proc-data", "stage3_with_llma.parquet")
    df.to_parquet(output_path)
    return output_path

# Define the Workflow
@workflow.defn
class AddressProcessingWorkflow:
    @workflow.run
    async def process_addresses(self) -> str:
        # Stage 1: Read Excel
        stage1_path = await workflow.execute_activity(
            read_excel,
            start_to_close_timeout=timedelta(seconds=60)
        )
        
        # Stage 2: Filter Data
        stage2_path = await workflow.execute_activity(
            filter_data,
            args=[stage1_path],
            start_to_close_timeout=timedelta(seconds=60)
        )
        
        # Stage 3: Add LLMA Column
        final_path = await workflow.execute_activity(
            add_llma_column,
            args=[stage2_path],
            start_to_close_timeout=timedelta(seconds=300)
        )
        
        return final_path

async def main():
    print('hello')
    client = await Client.connect("localhost:7233")
    
    # Register workflow and activities
    worker = Worker(
        client,
        task_queue="address-processing",
        workflows=[AddressProcessingWorkflow],
        activities=[read_excel, filter_data, add_llma_column]
    )
    
    async with worker:
        try:
            result = await client.execute_workflow(
                AddressProcessingWorkflow.process_addresses,
                id="address-processing-workflow",
                task_queue="address-processing",
            )
            print(f"Workflow completed. Final output at: {result}")
        except exceptions.WorkflowAlreadyStartedError:
            print("Workflow is already running. Please wait for it to complete or terminate it.")
            return

if __name__ == "__main__":
    import asyncio
    print('main')
    asyncio.run(main())

