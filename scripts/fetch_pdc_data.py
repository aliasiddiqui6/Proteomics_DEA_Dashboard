import requests
import json
import pandas as pd

def fetch_cptac_clinical_data(pdc_study_id):
    """
    Fetches clinical metadata for a given PDC Study ID via GraphQL.
    """
    url = 'https://pdc.cancer.gov/graphql'
    
    # GraphQL query to request specific clinical fields
    query = f'''{{
        clinicalDataPerStudy(study_id: "{pdc_study_id}", acceptDUA: true) {{
            case_submitter_id
            disease_type
            primary_site
            ethnicity
            sex
            tumor_stage
        }}
    }}'''
    
    # Send the POST request
    response = requests.post(url, json={'query': query})
    
    if response.status_code == 200:
        data = response.json()
        # Parse the JSON response into a pandas DataFrame
        clinical_records = data['data']['clinicalDataPerStudy']
        df = pd.DataFrame(clinical_records)
        return df
    else:
        raise Exception(f"API request failed with status code {response.status_code}.")

# Example execution using a placeholder PDC study ID
if __name__ == "__main__":
    study_id = "PDC000153" 
    print(f"Fetching data for Study ID: {study_id}...")
    
    df_clinical = fetch_cptac_clinical_data(study_id)
    
    # Save directly to the raw data folder
    output_path = "../data/raw/api_fetched_clinical_data.csv"
    df_clinical.to_csv(output_path, index=False)
    print(f"Data saved successfully to {output_path}")