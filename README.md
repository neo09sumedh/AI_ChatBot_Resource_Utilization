# AWS Cloud Cost Optimization Chatbot 🤖

A scalable, serverless chatbot pipeline built using AWS services and React.js to analyze and optimize cloud resource usage based on user queries. The system captures user input, interprets it using Amazon Lex, invokes relevant Lambda workflows, and leverages Amazon SageMaker for predictive analytics, returning results via DynamoDB.

## 🏗️ AWS Architecture
![image](https://github.com/user-attachments/assets/7b4ee4b2-fa21-444d-bf41-c2b59adda69e)



## 🚀 Features

* **Serverless Architecture**: Cost-effective, on-demand usage without maintaining servers
* **Natural Language Processing**: Amazon Lex for understanding user queries
* **Real-time Recommendations**: EC2 instance rightsizing suggestions via SageMaker
* **Asynchronous Processing**: Query handling through Amazon SQS and Step Functions
* **Scalable Data Management**: Efficient processing using Lambda and DynamoDB
* **Request Tracking**: Query status tracking using unique request/session IDs

## 🧠 Architecture Overview

The chatbot follows a sophisticated serverless architecture pattern:

1. User submits query through React frontend
2. API Gateway routes request to Lambda
3. Lambda forwards to Amazon Lex for intent recognition
4. Structured data flows to SQS for asynchronous processing
5. Step Functions orchestrate analysis workflows
6. SageMaker provides predictive insights
7. Results stored in DynamoDB for retrieval

## 🛠️ Tech Stack

### 💻 Frontend
* React.js
* HTML, CSS, JavaScript

### ☁️ Cloud Services
* AWS Lambda
* Amazon API Gateway
* Amazon Lex v2
* Amazon SQS
* Amazon SageMaker
* Amazon CloudWatch
* AWS Cost Explorer
* AWS Step Functions
* Amazon S3
* Amazon DynamoDB

### 🔧 Languages & Tools
* Python (Boto3, Regex)
* Postman (for testing APIs)
* Git & GitHub

## 🖥️ Frontend Interface (React)

The chatbot interface was built using React.js to enable a clean and interactive frontend for querying AWS resource usage.

### 📁 Key Files
* **App.js**: Main component that handles user interaction and sends requests
* **index.js**: Application entry point
* **App.css, index.css**: Stylesheets for component and global styling
* **reportWebVitals.js**: Performance monitoring (optional)
* **setupTests.js, App.test.js**: Set up and define component testing

### 🚀 Frontend Features
* Clean UI to interact with the Amazon Lex chatbot
* Users can type natural language queries like:
  * "Check my EC2 usage from Jan 1 to Jan 15"
  * "Is my instance i-08fabc123xyz underutilized?"
* The frontend sends queries to API Gateway, which forwards them to Lambda for Lex processing

## 📥 AWS Lex Input Intents

### 1. CheckAWSUsage
* **Slots**: service_name, from_date, to_date
* **Example Query**: "Check my EC2 usage from 2024-01-01 to 2024-02-01"
* **Features**: Date parsing (via custom slot + regex) and service identification

### 2. CheckInstanceSize
* **Slots**: instance_id
* **Example Query**: "Is my EC2 instance i-123abc oversized?"

## 🧩 Lambda Functions

### 1. APIToLexHandler
* Triggered by API Gateway
* Receives user input, forwards it to Amazon Lex using recognizeText()

### 2. LexToSQSHandler
* Fulfillment Lambda connected to Lex
* Extracts intent and slot values
* Formats message and pushes to Amazon SQS for asynchronous processing

### 3. LambdaSqsStepFunction
* Triggered by SQS events
* Starts a Step Function execution using the received message to orchestrate downstream ML/data tasks

### 4. OtherServicesUtilization
* Fetches historical usage data from AWS CloudWatch and Cost Explorer
* Aggregates and analyzes metrics for services like S3, Lambda, RDS, etc.
* Formats and stores the result in DynamoDB with cost and utilization summary

### 5. SageMaker Predictor Lambda
* Retrieves EC2 metrics from CloudWatch and other sources
* Sends data to a deployed Random Forest model on SageMaker
* Generates instance type recommendations and stores them in DynamoDB

### 6. GET Lambda (Status Retrieval)
* Triggered via API GET request
* Queries DynamoDB using session ID to retrieve processed results for the user

## 📊 SageMaker Model (CheckInstanceSize Intent)

* **Input**: JSON-formatted vector of historical EC2 usage metrics (e.g., CPUUtilization, Network I/O, Disk Ops) associated with an instance_id
* **Model**: Trained Random Forest classifier designed to evaluate resource utilization patterns and classify instance efficiency
* **Output**: Label indicating whether the instance is Underutilized, Overutilized, or Right-Sized, enabling actionable rightsizing recommendations

## 💬 Example Bot Interactions

Here are some example interactions with the chatbot:

```typescript
// Example 1: SQS Message Processing
Input: "Check my SQS queue metrics"
Output: "Service: SQS, Cost: 0.00 USD, Utilization Summary: SentMessageSize: 968.7351190476619, NumberOfMessagesSent: 0.11980669014747812, ApproximateNumberOfMessagesDelayed: 0.0"

// Example 2: Lambda Function Analysis
Input: "Can you get my Lambda usage from 2025-04-01 to 2025-04-15?"
Output: "Service: Lambda, Cost: 0.00 USD, Utilization Summary: Errors: 0.0, ConcurrentExecutions: 3.0, Duration: 2812.865"

// Example 3: EC2 Instance Optimization
Input: "Is my EC2 instance i-0efd4ce1cbdccabb0 right-sized?"
Output: "Scale down to a smaller instance to reduce costs. Suggested type: t2.xlarge"
```


## 🚀 Running the Application
### Prerequisite 
1) AWS CLI configured with appropriate permissions (IAM role with access to Lambda, Lex, SQS, DynamoDB, Step Functions, SageMaker, API Gateway, S3, Cost Explorer, CloudWatch)
2) Node.js / Python installed locally (for Lambda functions)
3) AWS CDK, SAM CLI, or Terraform (optional for IaC deployments)
4) Amazon Lex Bot configured with intents for cost queries
5) Need to deploy AWS services manually , Terraform deployment is planned in next Phase.
### Development Setup

```bash
# Install dependencies
npm install

# Start the development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

1️⃣ Clone the Repository

   <pre> git clone https://github.com/neo09sumedh/AI_ChatBot_Resource_Utilization.git
   cd AI_ChatBot_Resource_Utilization</pre>

2️⃣ Set Up S3 Bucket for Historical Data
Upload historical Cost & Usage Reports and CloudWatch metrics to an S3 bucket.

 <pre> aws s3 mb s3://<your-s3-bucket-name>
aws s3 cp historical_data/ s3://<your-s3-bucket-name>/ --recursive </pre>

3️⃣ Deploy DynamoDB Table
Provision a DynamoDB table to store chat request/response metadata.

 <pre> aws dynamodb create-table \
    --table-name ChatbotRequests \
    --attribute-definitions AttributeName=SessionId,AttributeType=S \
    --key-schema AttributeName=SessionId,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST </pre> 

4️⃣ Deploy Lambda Functions
Deploy the following Lambda functions:
Amazon Lex Invocation Lambda : APIToLexHandler.py
Publisher Lambda : LexToSQSHandler.py
Subscriber Lambda : LambdaSqsStepFunction.py
Utilization Explorer Lambda : OtherServicesUtilization.py
Model Predictor Lambda : InvokeCloudUtilizationStepFunction.json
Response Lambda : fetch_response.py

You can deploy each function manually or automate using AWS SAM/CDK.

Example (manual deploy):

 <pre> 
cd lambda_functions/lex_invocation
zip function.zip *
aws lambda create-function --function-name LexInvocationLambda \
    --runtime python3.11 --role <IAM-role-ARN> \
    --handler app.handler --zip-file fileb://function.zip
Repeat for other Lambda functions.

   </pre> 

5️⃣ Create Amazon Lex Bot
Configure Amazon Lex with intents for:
Cost Explorer queries
Utilization insights
Forecast queries
Ensure LexInvocationLambda is configured as the Lambda fulfillment for intents.

6️⃣ Configure SQS Queue
Create an SQS queue for message orchestration.

 <pre> aws sqs create-queue --queue-name ChatbotMessageQueue </pre> 

7️⃣ Deploy Step Functions Workflow
Create a Step Functions state machine to orchestrate:

Utilization Explorer Lambda

Model Predictor Lambda

Training Data Preparation Lambda

Example (if using AWS Console JSON definition):
Use ASL (Amazon States Language) to define the workflow and link Lambda ARNs.

8️⃣ Configure API Gateway Endpoints
Create Request API and Response API endpoints to integrate with:

LexInvocationLambda (Request API)

ResponseLambda (Response API)

Example:
 <pre> aws apigateway create-rest-api --name 'ChatbotRequestAPI' </pre> 

Deploy APIs and note down the invoke URLs.

9️⃣ Set Up SageMaker Endpoints
Train your model in SageMaker using historical data from S3 and deploy a real-time inference endpoint.
# Example SageMaker Notebook instance setup
# Upload and run your training script inside SageMaker Studio or Notebook instance

🔟 Integrate Frontend Application
Develop your application window using React / Java / Python
Connect to Request and Response APIs to enable user interaction.

## 📡 API Interaction

You can interact with the chatbot in two ways:

### 1. Using the Chatbot UI

Simply type your message in the chat interface. For example:
```
Check my EC2 usage from 2025-01-01 to 2025-04-04
```

### 2. Using Postman or API Client


Headers:
```
Content-Type: application/json
```

Request Body:
```json
{
  "message": "Check my EC2 usage from 2025-01-01 to 2025-04-04"
}
```

Response:
```json
{
  "type": "bot",
  "content": "Processing your request...",
  "timestamp": "2025-03-20T10:30:00Z"
}
```

## 📸 Output Preview

 ![image](https://github.com/user-attachments/assets/b5320e3e-1433-42d5-85f6-a8df66666251)

 
## 🔮 Future Enhancements

* **Real-Time Notifications**: Integrate Amazon SNS to alert users once processing is complete, enhancing responsiveness and engagement
* **Multi-Service Reporting**: Enable support for comparing and reporting usage across multiple AWS services in a single query
* **Expand ML Model Capabilities**: Extend the SageMaker model to support additional AWS services and multi-metric analysis for more precise rightsizing and cost-optimization recommendations

## 🙌 Authors & Contributions

* **Aryan Agrawal** – Developed the pipeline from API Gateway to SQS using Lex and Lambda for capturing user queries and forwarding structured messages
* **Sumedh Ambapkar** – Designed the backend architecture and implemented the SageMaker integration; contributed to the React frontend development
* **Arju Singh** – Managed the flow from SQS to Step Functions, triggering the appropriate Lambda functions for async processing
* **Krishna** – Implemented logic to fetch service cost and utilization metrics using CloudWatch and AWS Cost Explorer
