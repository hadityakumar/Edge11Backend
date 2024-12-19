# **Python Application**

This project provides a Python-based API application using **FastAPI**. Follow the steps below to set up and run the project locally.

---

## **Prerequisites**
Before proceeding, ensure that you have the following installed on your system:

- **Python** (>=3.8)
- **pip** (Python package manager)
- **Git**

---

## **Setup Instructions**

### **1. Clone the Repository**
To get started, clone this repository to your local machine using the following command:

```bash
git clone <repository-url>
```

### **2. Navigate to the Project Directory**
Go to the project directory where the repository is cloned:

```bash
cd <repository-folder>
```

### **3. Install Dependencies**
Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### **Configuration Files**
The following configuration files are required for the application:

- **.env**: Contains sensitive environment variables, like API keys or database credentials.
    - You need to create a `.env` file in the project root and add your OpenAI API key like so:
    ```env
    OPENAI_APIKEY=your-openai-api-key
    ```
---

### **Environment Variables**
The application uses the following environment variables:
- `OPENAI_APIKEY`: Your OpenAI API key to access OpenAI's services.

Make sure you have set these variables correctly in the `.env` file before starting the application.


### **Run the Application**
Start the FastAPI application locally using the `uvicorn` server:

```bash
python -m uvicorn main:app --reload
```


## **API Endpoints**

Below are the available API endpoints and their descriptions:

---

### **1. Predict**
- **Endpoint**: `POST /predict`
- **Path**: `routes/predict.py`
- **Description**: Handles predictions based on the provided input.

---

### **2. Train and Test**
- **Endpoints**:
  - `POST /train_and_test`: Trains and tests the model.
  - `GET /`: Returns metadata or status.
- **Path**: `routes/model.py`

---

### **3. Get Players**
- **Endpoint**: `GET /players`
- **Path**: `routes/getplayers.py`
- **Description**: Fetches details of players.

---

### **4. Get Unique Teams**
- **Endpoint**: `GET /unique_teams`
- **Path**: `routes/getteams.py`
- **Description**: Returns a list of unique teams.

---

### **5. Chatbot**
- **Endpoint**: `POST /chatbot`
- **Path**: `routes/chatbot.py`
- **Description**: Handles chatbot interactions.


## **Project Structure**
The application follows a modular structure:

bash
```
project-root/
│
├── main.py                # Entry point of the application
├── routes/                # Contains all route modules
│   ├── predict.py
│   ├── model.py
│   ├── getplayers.py
│   ├── getteams.py
│   ├── chatbot.py
│
├── requirements.txt       # Dependencies
├── README.md              # Documentation
└── ...

```