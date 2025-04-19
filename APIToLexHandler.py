import json
import boto3

# Initialize AWS Lex client
lex = boto3.client("lexv2-runtime")

# Lex Bot Details
LEX_BOT_ID = "2C5KLYWSCK"
LEX_ALIAS_ID = "TSTALIASID"

def lambda_handler(event, context):
    try:
        print("Event received: ", json.dumps(event))
        # Step 1: Parse incoming request
        body = json.loads(event["body"]) if "body" in event else event  
        user_message = body.get("message", "")
        sessionId=body.get("sessionId", "")

        # Send user input to Lex
        lex_response = lex.recognize_text(
            botId=LEX_BOT_ID,
            botAliasId=LEX_ALIAS_ID,
            localeId="en_US",
            #sessionId="user-session",
            sessionId=sessionId,
            text=user_message
        )

        # Return Lex's response back to API Gateway
        return {
            "statusCode": 200,
            "body": json.dumps({"LexResponse": lex_response})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
