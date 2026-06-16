

import os
import tempfile
import fitz
import pytesseract
import gspread
import pandas as pd
import streamlit as st

from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel
from oauth2client.service_account import ServiceAccountCredentials

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Hotel Receipt AI Extractor",
    layout="wide"
)

st.title("🏨 Hotel Receipt AI Extractor")

# ==================================================
# LOAD ENV
# ==================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env")
    st.stop()

# ==================================================
# PYDANTIC MODEL
# ==================================================

class BookingDetails(BaseModel):

    guest_name: str | None = None
    hotel_name: str | None = None
    confirmation_id: str | None = None
    check_in_date: str | None = None
    check_out_date: str | None = None

parser = PydanticOutputParser(
    pydantic_object=BookingDetails
)

# ==================================================
# PROMPT
# ==================================================

prompt = PromptTemplate(
    template="""
You are an expert hotel booking receipt extractor.

Extract hotel booking information ONLY from the hotel PDF text.

Required Fields:
- guest_name
- hotel_name
- confirmation_id
- check_in_date
- check_out_date

Instructions:

- Hotel PDFs can have any layout
- Never use filename
- Never guess
- If missing return null

{format_instructions}

HOTEL PDF TEXT:
{text}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions":
        parser.get_format_instructions()
    }
)

# ==================================================
# LLM
# ==================================================

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)

chain = prompt | llm | parser

# ==================================================
# GOOGLE SHEETS
# ==================================================

def connect_google_sheet():

    try:

        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]

        creds = ServiceAccountCredentials.from_json_keyfile_name(
            "credentials.json",
            scope
        )

        client = gspread.authorize(creds)

        sheet = client.open_by_url(
            "https://docs.google.com/spreadsheets/d/1PNtZvdpU-91QrZWcmnBYFXzZhsNhx5vJIhb-A7qopoc/edit?usp=sharing"
        ).sheet1

        return sheet

    except Exception as e:

        st.error(f"Google Sheet Error: {e}")
        return None

# ==================================================
# OCR FUNCTION
# ==================================================

def extract_text_from_pdf(pdf_path):

    full_text = ""

    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):

        page = pdf_document[page_num]

        pix = page.get_pixmap(
            matrix=fitz.Matrix(4, 4)
        )

        image_path = os.path.join(
            tempfile.gettempdir(),
            f"page_{page_num}.png"
        )

        pix.save(image_path)

        image = Image.open(image_path)

        text = pytesseract.image_to_string(
            image
        )

        full_text += text + "\n"

    return full_text

# ==================================================
# FILE UPLOAD
# ==================================================

uploaded_files = st.file_uploader(
    "Upload Hotel PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# ==================================================
# PROCESS
# ==================================================

if st.button("Process PDFs"):

    if not uploaded_files:

        st.warning("Upload at least one PDF")
        st.stop()

    results = []

    sheet = connect_google_sheet()

    progress = st.progress(0)

    for idx, uploaded_file in enumerate(uploaded_files):

        try:

            st.subheader(uploaded_file.name)

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(uploaded_file.read())

                pdf_path = tmp.name

            text = extract_text_from_pdf(
                pdf_path
            )

            with st.expander(
                f"Text Preview - {uploaded_file.name}"
            ):
                st.text(text[:2000])

            result = chain.invoke(
                {"text": text}
            )

            row = {
                # "file_name":
                # uploaded_file.name,

                "guest_name":
                result.guest_name,

                "hotel_name":
                result.hotel_name,

                "confirmation_id":
                result.confirmation_id,

                "check_in_date":
                result.check_in_date,

                "check_out_date":
                result.check_out_date
            }

            results.append(row)

            if sheet:

                sheet.append_row([
                    # uploaded_file.name,
                    result.guest_name,
                    result.hotel_name,
                    result.confirmation_id,
                    result.check_in_date,
                    result.check_out_date
                ])

            st.success(
                f"Processed {uploaded_file.name}"
            )

        except Exception as e:

            st.error(
                f"{uploaded_file.name} : {str(e)}"
            )

        progress.progress(
            (idx + 1) / len(uploaded_files)
        )

    if results:

        df = pd.DataFrame(results)

        st.subheader("Extracted Results")

        st.dataframe(
            df,
            use_container_width=True
        )

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download CSV",
            csv,
            "hotel_results.csv",
            "text/csv"
        )
    