import json
import boto3
import uuid
import re

# Initialize AWS SQS client
sqs_client = boto3.client('sqs')

# Replace with your actual SQS Queue URL
QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/324037300355/LexOutputQueue"

def lambda_handler(event, context):
    try:
        print("Event received: ", json.dumps(event))

        # Generate a unique request ID
        request_id = str(uuid.uuid4())

        # Check if Lex detected an intent
        interpretations = event.get('interpretations', [])
        if not interpretations:
            return {
                "sessionState": {
                    "dialogAction": {"type": "Close"},
                    "intent": {
                        "name": "UnknownIntent",
                        "state": "Failed"
                    }
                },
                "messages": [{
                    "contentType": "PlainText",
                    "content": "No intent recognized. Please try again."
                }]
            }

        # Extract intent details
        intent_request = event["sessionState"]["intent"]
        intent_name = intent_request["name"]
        slots = intent_request.get("slots", {})

        # Get the original user query
        user_query = event.get("inputTranscript", "Unknown query")
        SessionId = event.get("sessionId", "Unknown SessionId")
        # Construct the message body
        message_body = {
            "request_id": request_id,
            "session_id": SessionId,
            "user_query": user_query,
            "intent_name": intent_name
        }

        # Extract slot values based on intent
        if intent_name == "CheckAWSUsage":
            # Extract service_name and date_range slot
            service_name = slots.get("service_name", {}).get("value", {}).get("interpretedValue", "unknown")
            #date_range = slots.get("date_range", {}).get("value", {}).get("interpretedValue", "unknown")
            from_date = slots.get("from_date", {}).get("value", {}).get("interpretedValue", "unknown")
            to_date = slots.get("to_date", {}).get("value", {}).get("interpretedValue", "unknown")


            # Print the extracted values
            print(f"Extracted values: service_name={service_name}, from_date={from_date},to_date={to_date}")
            
            # Parse the date_range to extract from_date and to_date
            #from_date, to_date = extract_dates_from_range(date_range)
            

            # Print the parsed dates
            print(f"Parsed dates: from_date={from_date}, to_date={to_date}")

            # Update the message body with all the extracted slot values
            message_body.update({
                "service_name": service_name,
                "from_date": from_date,
                "to_date": to_date
            })
        elif intent_name == "CheckInstanceSize":
            message_body.update({
                "instance_id": slots.get("instance_id", {}).get("value", {}).get("interpretedValue", "unknown")
            })

        # Send message to SQS and capture response
        sqs_response = sqs_client.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(message_body)
        )

        # Return Lex-compatible response **(without sessionAttributes)**
        return {
            "sessionState": {
                "dialogAction": {"type": "Close"},
                "intent": {
                    "name": intent_name,
                    "state": "Fulfilled"
                }
            },
            "messages": [{
                "contentType": "PlainText",
                "content": f"Your request has been received. Use request ID: {request_id} to track its status."
            }]
        }

    except Exception as e:
        return {
            "sessionState": {
                "dialogAction": {"type": "Close"},
                "intent": {
                    "name": intent_name if 'intent_name' in locals() else "UnknownIntent",
                    "state": "Failed"
                }
            },
            "messages": [{
                "contentType": "PlainText",
                "content": f"Error: {str(e)}"
            }]
        }


# Function to extract from_date and to_date from date_range string
def extract_dates_from_range(date_range):
    # Using regex to match date range formats like "from <date> to <date>", "between <date> and <date>", etc.
    date_pattern = r"(from|between)\s*(\d{4}-\d{2}-\d{2})\s*(to|and)\s*(\d{4}-\d{2}-\d{2})"
    match = re.search(date_pattern, date_range, re.IGNORECASE)

    if match:
        from_date = match.group(2)
        to_date = match.group(4)
    else:
        from_date = "unknown"
        to_date = "unknown"

    return from_date, to_date
