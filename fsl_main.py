#!/usr/bin/env python
# coding: utf-8
# import the libraries
import os
import pandas as pd
import json
import argparse
from few_shot_learning import FSLModel

# use argparse to let the user provides values for variables at runtime
def DataTestingArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', 
        type=str, required=True, help='Please specify the task: SA or MCQA?')
    parser.add_argument('--language', 
        type=str, required=True, help='Please specify the language')
    parser.add_argument('--model_name', 
        type=str, required=True, help='Load a LLM as model for few-shot learning')
    parser.add_argument('--dataset_dir', 
        type=str, required=True, help='Directory of the dataset files')
    parser.add_argument('--batch_size', 
        type=int, default=10, help='Number of request sentences per batch of target translation')
    parser.add_argument('--save_dir', 
        type=str, required=True, help='Directory for saving experimental results')
    args = parser.parse_args()
    return args

def main():
    #create the configuration class
    args=DataTestingArguments() #call the arguments
    
    ####---Load Dataset---####
    train_data = pd.read_csv(os.path.join(args.dataset_dir, f"data_train_{args.task}_{args.language}.csv"))
    test_data = pd.read_csv(os.path.join(args.dataset_dir, f"data_test_{args.task}_{args.language}.csv"))
   
    #test_data = test_data [:4]
    if args.task == 'SA':
        test_data = test_data[['review']]
        train_data = train_data[['review', 'label']]
    else:
        test_data = test_data[[f'question_{args.language}', f'choices_all_{args.language}']] # -- question_ladin && choices_all_ladin --
        train_data = train_data[[f'question_{args.language}', f'choices_all_{args.language}', 'answer']] # -- question_ladin && choices_all_ladin --

    # Translation loop with rotating few-shot examples
    rotation_index = 0  # Initialize a cumulative rotation index

    # Set the LLMs
    fsl_model = FSLModel(args.model_name)
    index_batch = 0
    
    for batch_start in range(0, len(test_data), args.batch_size):
        # Get the batch of test data
        test_batch = test_data.iloc[batch_start:batch_start + args.batch_size]
        
        # Select a rotating subset of few-shot examples
        few_shot_examples = train_data.iloc[rotation_index:rotation_index + (args.batch_size*2)] 

        # Wrap around if reaching the end of train_data
        if rotation_index + (args.batch_size*2) >= len(train_data):
            rotation_index = (rotation_index + (args.batch_size*2)) % len(train_data)
        else:
            rotation_index += (args.batch_size*2) # Increment for the next batch
        
        if args.task == 'SA':
            # Convert DataFrame to a list in the desired format
            formatted_test_batch = [f"[review: {row['review']}, label: ]" for _, row in test_batch.iterrows()]
            formatted_few_shot_examples = [f"[review: {row['review']}, label: {row['label']}]" for _, row in few_shot_examples.iterrows()]
            # make it to string
            formatted_test_batch = ", ".join(formatted_test_batch)
            formatted_few_shot_examples = ", ".join(formatted_few_shot_examples)

            # Prompt templates
            prompt_1 = (f"Below are Tripadvisor reviews in {args.language} along with their sentiment labels:\n\n" # Ladin (Val Badia variant)
                        )
            prompt_2 = (
                    f"\nPlease classify the sentiment for the following {len(test_batch)} Tripadvisor reviews in {args.language} as either 0 (Positive) or 1 (Negative). " ## Ladin (Val Badia variant)
                    "Fill in the empty 'label' fields with only 0 or 1. Respond with the sentiment labels in list format like this: [x, x, x, ..., x]. "
                    "Do not include any additional explanations or text.\n"
                )
        else:
            # Convert DataFrame to a list in the desired format

            # -- question_ladin && choices_all_ladin --

            formatted_test_batch = [f"[question: {row[f'question_{args.language}']}, choices: {row[f'choices_all_{args.language}']}, answer: ]" for _, row in test_batch.iterrows()] 
            formatted_few_shot_examples = [f"[question: {row[f'question_{args.language}']}, choices: {row[f'choices_all_{args.language}']}, answer: {row['answer']}]" for _, row in few_shot_examples.iterrows()]
            # make it to string
            formatted_test_batch = ", ".join(formatted_test_batch)
            formatted_few_shot_examples = ", ".join(formatted_few_shot_examples)

            # Prompt templates
            prompt_1 = (f"Below are multiple-choice questions in {args.language} with 3, 4, or 5 answer choices. " ## Ladin (Val Badia variant)
                        "The correct answer is explicitly provided as a id number corresponding to the order of the choices:\n\n")

            prompt_2 = (f"Please answer the questions based on the available choices in {args.language}, by filling in the empty 'answer' fields with the id number corresponding to the order of the choices. "
                        "Provide the answers in a list format like this: [x, x, x, ..., x]. " ## Ladin (Val Badia variant)
                        "Do not include any additional explanations or text.\n")

            
        # Generate translation using LLMs with API
        generated_translation = fsl_model.generating(prompt_1, prompt_2, formatted_few_shot_examples, formatted_test_batch)
        
        # If the response is successful
        #if generated_translation.status_code == 200:
        try:
            response_json = generated_translation.json()
            print(f"Response JSON for batch {index_batch} ")

            # prepraing set the output into json file
            if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
            output_path = os.path.join(args.save_dir, f'{args.task}_{args.language}_{args.model_name}_size of_{args.batch_size}_batch_{index_batch}.json') # _italian
                
            # Save the JSON response for this batch
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(response_json, json_file, ensure_ascii=False, indent=4)
                
            print(f"Saved batch {index_batch} to {output_path}")
        except (json.JSONDecodeError, KeyError) as e:
            print("Error parsing the translation output.")
                
        #else:
        #    print(f"Error: {generated_translation.status_code}, {generated_translation.text}")

        index_batch += 1    
if __name__ == "__main__":
    main()
