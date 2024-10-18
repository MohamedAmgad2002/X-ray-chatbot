import getpass
import os
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable.config import RunnableConfig
from preprocessing import retina_preprocessing, pneumonia_preprocessing, kidney_brain_preprocessing, general_preprocessing, markdown_to_text
import tensorflow as tf
from text_to_speech import tts
import edge_tts
import chainlit as cl
from chainlit.input_widget import TextInput

# Load your Google API key from an environment variable
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass(
        "Enter your Google AI API key: ")

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# latest message send by model
latest_message = ""
msg_id = None

# prompt template
baymax_template = """
You are Baymax, a helpful medical doctor. Provide advice for managing the following conditions until the user can see a doctor: Diabetic Retinopathy, Pneumonia, Brain Tumors, Kidney diseases (Cysts, Tumors, Stones). After describing the condition, provide at least 3 specific and practical steps or lifestyle adjustments that the user can take. Make the advice actionable.

If asked about something not on the list, reply with: "I don't know."

Chat History: {chat_history}
User's Question: {question}
Baymax:
"""

diagnose_template = """
You are a medical assistant that provides detailed, clear, and empathetic responses based on the medical condition provided.

Condition: {condition}

Respond with a structured explanation of the diagnosis. Begin by stating the findings from the scan or medical test, describing the severity, and then follow with treatment recommendations or next steps.

Here's the format:
1. **Findings from the scan/test**: Clearly state what the scan or test revealed.
2. **Explanation of the condition**: Provide a brief explanation of the condition and its severity.
3. **Next steps or treatments**: Suggest possible treatment options or steps to manage the condition.

Conclude the response with the following: "Feel free to ask any questions you may have."

Response:
- Start with: "The scan reveals {condition}. This suggests..."
- Include an explanation of the condition.
- End with: "Feel free to ask any questions you may have."
"""





# prompt
baymax_prompt = PromptTemplate(
    input_variables=['chat_history', 'question'],
    template=baymax_template
)


diagnose_prompt = PromptTemplate(
    input_variables=['condition'],
    template=diagnose_template
)

def which_model(image_path):
    """
    Predicts which model to use for a given image.

    Args:
        image_path: str, the path to the image file

    Returns:
        int, the index of the model to use
    """
    model = tf.keras.models.load_model('models/general.h5')

    image = general_preprocessing(image_path)

    prediction = model.predict(image)
    return np.argmax(prediction)

# preprocess input


def preprocess(image_path, model_idx):
    """
    Preprocesses an image according to the model index.

    Args:
        image_path: str, the path to the image file
        model_idx: int, the index of the model to use

    Returns:
        tensor, the preprocessed image
    """
    if model_idx == 0 or model_idx == 1:
        image = kidney_brain_preprocessing(image_path)
    elif model_idx == 2:
        image = pneumonia_preprocessing(image_path)
    else:
        image = retina_preprocessing(image_path)
    return image


def encoder(model_idx):
    """
    Encodes the model index into a dictionary based on the type of model.

    Parameters:
        model_idx (int): The index of the model.

    Returns:
        dict: A dictionary mapping the model index to its corresponding model name.
    """
    if model_idx == 0:
        encoder = kidney_brain_encoder = {
            0: "Brain Tumor",
            1: "Healthy Brain",
            2: "Kidney Cyst",
            3: "Kidney Normal",
            4: "Kidney Stone",
            5: "Kidney Tumor"
        }
    elif model_idx == 2:
        encoder = chest_encoder = {
            0: 'Chest Normal',
            1: 'Chest Pneumonia',
        }
    else:
        encoder = retina_encoder = {
            0: 'Diabetic Retinopathy',
            1: 'No Diabetic Retinopathy',
        }
    return encoder


def predict(image_path):
    """
    Predicts the disease based on the input image path.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        str: The predicted disease based on the input image.
    """
    model_idx = which_model(image_path)

    image = preprocess(image_path, model_idx)

    if model_idx == 0 or model_idx == 1:
        model = tf.keras.models.load_model('models/Kidney_brain_model.h5')
        encoder_dic = encoder(0)
        prediction = model.predict(image)
        result = encoder_dic[np.argmax(prediction)]
    elif model_idx == 2:
        model = tf.keras.models.load_model('models/chest.h5')
        encoder_dic = encoder(2)
        prediction = model.predict(image)
        result = encoder_dic[0 if prediction < 0.5 else 1]

    else:
        model = tf.keras.models.load_model('models/retina_model.h5')
        encoder_dic = encoder(3)
        prediction = model.predict(image)
        result = encoder_dic[0 if prediction < 0.5 else 1]

    return result


@cl.action_callback("audio")
async def on_action(action: cl.Action):
    global latest_message
    latest_message = markdown_to_text(latest_message)
    audio = tts(latest_message)
    await audio.save('message.mp3')
    # Send the audio file as a response using content and link to the original message using for_id
    await cl.Audio(path ='message.mp3',filename="baymax_response.mp3", auto_play=True).send(for_id=msg_id)


@cl.on_chat_start
async def session_start():
    global msg_id, latest_message

    # audio button action
    actions = [
        cl.Action(name="audio", value='audio',description="read aloud")
    ]

    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest')

    # Create history
    conversation_memory = ConversationBufferMemory(memory_key="chat_history",
                                                   max_len=50,
                                                   return_messages=True,
                                                   )
    baymax_chain = LLMChain(llm=llm, prompt=baymax_prompt,
                         memory=conversation_memory)
    
    diagnose_chain = LLMChain(llm=llm, prompt=diagnose_prompt, memory=conversation_memory)

    # save chain to session
    cl.user_session.set('llm_chain', baymax_chain)
    cl.user_session.set('action', actions)


    images = None

    # Wait for the user to upload a file
    while images == None:
        images = await cl.AskFileMessage(
            content="Hello, I am `Baymax` your personal Healthcare assistant\nplease upload your Scan to start Chatting",
            accept=["image/jpg", "image/jpeg", "image/png"]
        ).send()

    image = images[0]
    image = cl.Image(path=image.path, name=image.name, display='inline')
    # Let the user know that the system is ready
    await cl.Message(
        content=f"Your Scan uploaded, please wait while I process it 😊",
        elements=[image]
    ).send()

    prediction = predict(image.path)

    response = await diagnose_chain.acall(prediction,
                                    callbacks=[
                                        cl.AsyncLangchainCallbackHandler()])
    
    msg = cl.Message(response["text"], actions=actions)
    latest_message = response['text']
    msg_id = msg.id

    await msg.send()



@cl.on_message
async def on_message(message: cl.Message):
    global latest_message, msg_id

    actions = cl.user_session.get('action')

    # load session variables
    llm_chain = cl.user_session.get('llm_chain')

    response = await llm_chain.acall(message.content,
                                     callbacks=[
                                         cl.AsyncLangchainCallbackHandler()])
    msg =  cl.Message(response["text"], actions=actions)
    
    latest_message = response['text']
    msg_id = msg.id

    await msg.send()