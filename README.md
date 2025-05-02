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
