from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
import re

# ==========================================================================================

model = ChatOpenAI(temperature=0.8, api_key="lm-studio", base_url="http://localhost:1233/v1")

# ==========================================================================================


class Bias(BaseModel):
    description: str = Field(default=None, description="Description of the bias")
    rating: int = Field(default=0, description="Bias rating (1-100)")

class BiasFilters(BaseModel):
    gender_bias: Bias = Field(default=None, description="Details about gender bias.")
    racial_bias: Bias = Field(default=None, description="Details about racial bias")
    age_bias: Bias = Field(default=None, description="Details about age bias")
    disability_bias: Bias = Field(default=None, description="Details about disability bias")
    socioeconomic_bias: Bias = Field(default=None, description="Details about socioeconomic bias")

class VideoKnowledgeRatings(BaseModel):
    clarity: Bias = Field(default=None, description="Measures how clearly technical concepts are explained")
    depth: Bias = Field(default=None, description="Assesses the thoroughness and depth of technical content")
    accuracy: Bias = Field(default=None, description="Evaluates the correctness of the information provided")

class PresentationSkills(BaseModel):
    engagement: Bias = Field(default=None, description="Assesses how engaging and interesting the video presentation is")
    organization: Bias = Field(default=None, description="Measures how well-organized and structured the content is")
    visual_aids: Bias = Field(default=None, description="Evaluates the effectiveness of visual aids in enhancing understanding")

class PracticalApplication(BaseModel):
    relevance: Bias = Field(default=None, description="Assesses how relevant the content is to real-world applications")
    examples: Bias = Field(default=None, description="Measures the quality and clarity of examples provided")
    hands_on: Bias = Field(default=None, description="Evaluates the inclusion of hands-on activities or demonstrations")

class VideoKnowledge(BaseModel):
    technical_knowledge: VideoKnowledgeRatings = Field(default=None, description="Technical knowledge ratings")
    presentation_skills: PresentationSkills = Field(default=None, description="Presentation skills ratings")
    practical_application: PracticalApplication = Field(default=None, description="Practical application ratings")

class BiasAndKnowledgeRatings(BaseModel):
    bias_filters: BiasFilters = Field(default=None, description="Bias filters and their ratings")
    video_knowledge_ratings: VideoKnowledge = Field(default=None, description="Ratings for video knowledge")


# ==========================================================================================


def analyze_text_for_bias(text_chunk):
    try:
        parser = JsonOutputParser(pydantic_object=BiasAndKnowledgeRatings)
        prompt_text = f'''
        Analyze the given text and identify instances of bias. Rate each type of bias (1-100) and provide descriptions for each bias category. Also, evaluate the content based on various video knowledge aspects and provide ratings and descriptions for each aspect. Output the analysis in the following
        Text: {text_chunk}
        '''
        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | model | parser
        output = chain.invoke({"query": prompt_text})
        return output
    except Exception as e:
        print(e)

# ==========================================================================================

def clean_file(FILE):
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', FILE)  # Removes non-ASCII characters
    file = re.sub(r'�', ' ', cleaned_text)             # Removes � characters specifically
    print(type(file))
    return file


def split(FILE_PATH):
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        file = f.read()
    
    file = clean_file(file)
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=700,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(file)
    return chunks

# ==========================================================================================

def process_chunks(file_path):
    chunks = split(FILE_PATH=file_path)

    for chunk in chunks:
        results = analyze_text_for_bias(chunk)

    return results


file_path = "subtitles.txt"
res = process_chunks(file_path=file_path)
print(res)