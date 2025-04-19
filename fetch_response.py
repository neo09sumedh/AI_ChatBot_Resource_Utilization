import json
import boto3
from boto3.dynamodb.conditions import Key

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
table_name = 'CloudCostUtilizationResponse'  # Replace with your table name
table = dynamodb.Table(table_name)

def lambda_handler(event, context):
    # Get sessionId from query parameters or body
    #session_id = event.get('queryStringParameters', {}).get('sessionId')
    session_id = event.get('sessionId')
    print("Received event:", json.dumps(event))
    #session_id = "1"
    if not session_id:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Missing sessionId parameter'})
        }
    
    try:
        # Query DynamoDB using sessionId
        response = table.query(
            KeyConditionExpression=Key('session_id').eq(session_id)
        )
        items = response.get('Items', [])
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'session_id': session_id,
                'responses': items
            }),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'  # For cross-origin access from your frontend
            }
        }

    except Exception as e:
        print(f"Error querying DynamoDB: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Server error'})
        }

