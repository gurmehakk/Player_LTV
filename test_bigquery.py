from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd

from google.cloud import bigquery
import pandas as pd

def test_connection():
    """Test BigQuery connection using default credentials"""
    try:
        # Initialize client with default credentials
        client = bigquery.Client()
        project_id = client.project
        
        print(f"✅ Successfully connected to BigQuery project: {project_id}")
        
        # Test query to list datasets
        print("\n📋 Listing datasets in project:")
        datasets = list(client.list_datasets())
        
        if not datasets:
            print("No datasets found in project.")
            return
        
        for dataset in datasets:
            print(f"- {dataset.dataset_id}")
            
        # Get table schema
        try:
            table_ref = client.dataset('test_data').table('app_data')
            table = client.get_table(table_ref)
            
            print("\n📋 Table Schema:")
            print(f"Table: {table.project}.{table.dataset_id}.{table.table_id}")
            print(f"Number of rows: {table.num_rows:,}")
            print(f"Size: {table.num_bytes / (1024*1024):.2f} MB")
            print("\nColumns:")
            for field in table.schema:
                print(f"- {field.name}: {field.field_type}")
                
            # Get sample data to understand the structure
            sample_query = f"""
            SELECT * 
            FROM `{project_id}.test_data.app_data`
            WHERE 
                server_timestamp IS NOT NULL
                AND (gaid IS NOT NULL OR idfa IS NOT NULL OR android_id IS NOT NULL OR waid IS NOT NULL OR idfv IS NOT NULL)
            LIMIT 1
            """
            
            print("\n🔍 Sample row:")
            sample = client.query(sample_query).to_dataframe()
            print(sample.iloc[0].to_dict() if not sample.empty else "No sample data found")
            
        except Exception as e:
            print(f"⚠️  Error getting table schema: {str(e)}")
            raise
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you're authenticated with: gcloud auth application-default login")
        print("2. Verify your project ID is set correctly")
        print("3. Check if you have the BigQuery Job User and BigQuery Data Viewer roles")
        raise

if __name__ == "__main__":
    test_connection()