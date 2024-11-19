from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging
import asyncio
from dotenv import load_dotenv
from datetime import datetime
import json
import os
import random

from langchain.output_parsers import PydanticOutputParser, RetryOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage


class Role(BaseModel):
    name: str = Field(description="The name of the role, can be a person or a job/ role/ skill description")
    description: str = Field(description="The description of the singularity of the role")

class ListBestRoles(BaseModel):
    roles: List[Role] = Field(description="A list of roles that would be best for the user prompt")

class TracebackData(BaseModel):
    timestamp: str
    user_prompt: str
    roles: List[Role]
    layer_responses: List[Dict[str, Any]]
    final_response: str
    total_cost: float



class MoA:
    def __init__(
        self,
        models_layer: List[str] = ["mistral-small", "mistral-medium"],
        model_roles: str = "mistral-medium",
        aggregator_model: str = "mistral-medium-latest",
        nb_models_by_layer: int = 2,
        layer: int = 0,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        layer_strategy: str = 'role'
    ):
        load_dotenv()
        self.models_layer = models_layer
        self.model_roles = model_roles
        self.aggregator_model = aggregator_model
        self.nb_models_by_layer = nb_models_by_layer
        self.layer = layer
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.layer_strategy = layer_strategy
        
        # Adjust layer based on strategy
        if layer_strategy == 'model':
            self.nb_models_by_layer = len(models_layer)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        
        llm_args = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # Update LLM initializations to include callback manager
        self.llms_layer = []
        for model_layer in self.models_layer:
            try:
                self.llms_layer.append(MoA.get_llm_from_model_name(model_layer)(
                    model=model_layer,
                    **llm_args
                ))
                self.llms_layer[-1].name = model_layer
            except Exception as e:
                logging.info(f"Error with initializing {model_layer}: {e}")
        assert len(self.llms_layer) > 0, "No LLMs initialized"
                
        self.llm_aggregator = MoA.get_llm_from_model_name(aggregator_model)(
            model=aggregator_model,
            **llm_args
        )
        self.llm_aggregator.name = aggregator_model
        self.llm_roles = MoA.get_llm_from_model_name(model_roles)(
            model=model_roles,
            **llm_args
        )
        self.llm_roles.name = model_roles
        
        if self.layer_strategy == 'model':
            self.nb_models_by_layer = len(self.models_layer)

        
        # Add traceback initialization
        os.makedirs("./logs/moa", exist_ok=True)
        self.traceback_filename = f"./logs/moa/moa_traceback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.base_entry = None
    
    @staticmethod
    def get_llm_from_model_name(model_name: str):
        """Create a LangChain provider instance."""
        if model_name.startswith("gpt") or model_name.startswith("chatgpt"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI
        elif model_name.startswith("claude"):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic
        elif model_name.startswith("llama") or model_name.startswith("gemma"):
            from langchain_groq import ChatGroq
            return ChatGroq
        elif model_name.startswith("mixtral") or model_name.startswith("mistral") or model_name.startswith("ministral"):
            from langchain_mistralai import ChatMistralAI
            return ChatMistralAI
        else:
            raise ValueError(f"Model {model_name} not supported")
        
    async def log_trace(self, entry_type: str, data: Any, layer: int = None, model_name: str = None, cost: float = None):
        """Log a trace entry to the traceback file"""
        if self.base_entry is None:
            raise ValueError("base_entry not initialized. Call run() first")
            
        entry = {
            **self.base_entry,
            "type": entry_type,
            "timestamp": datetime.now().isoformat()
        }
        
        if layer is not None:
            entry["layer"] = layer
        if model_name is not None:
            entry["model_name"] = model_name
        if cost is not None:
            entry["cost"] = cost
            
        entry["data"] = data

        with open(self.traceback_filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def generate_roles(self, user_prompt: str, count: int = 5):
        parser = PydanticOutputParser(pydantic_object=ListBestRoles)
        retry_parser = RetryOutputParser.from_llm(parser=parser, llm=self.llm_roles)
        
        llm_prompt = ChatPromptTemplate.from_messages([
            ("system", "Give me a list of {count} roles that would be best for this user prompt"
             "You choose the role best of their ability to solve the user prompt based on their reasoning"
             "not because they got special knowledge or private information"
             "{format_instructions}"
            ),
            ("user", "{user_prompt}")
        ])
        
        chain = llm_prompt | self.llm_roles | parser
        main_chain = RunnableParallel(
            completion=chain, prompt_value=llm_prompt
        ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

        response = main_chain.invoke({
            "user_prompt": user_prompt,
            "count": count,
            "format_instructions": parser.get_format_instructions()
        })
        await self.log_trace("roles", response.model_dump(), model_name=self.model_roles)
        return response.model_dump()

    async def generate_based_on_roles(self, llm, user_prompt: str, name: str, description: str):
        llm_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a {name}. {description}"),
            ("user", "{user_prompt}")
        ])
        
        chain = llm_prompt | llm
        response = chain.invoke({
            "name": name,
            "description": description,
            "user_prompt": user_prompt
        })
        
        await self.log_trace(
            "role_response",
            {"role": name, "response": response.content},
            model_name=llm.name
        )
        return response.content
    
    async def generate(self, llm, user_prompt: str):
        llm_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            ("user", "{user_prompt}")
        ])
        
        chain = llm_prompt | llm
        response = chain.invoke({
            "user_prompt": user_prompt
        })
        
        await self.log_trace(
            "role_response",
            response.content,
            model_name=llm.name
        )
        return response.content
    

    async def generate_layer(self, user_prompt: str):
        responses = []
        import random
        if self.layer_strategy == 'role':
        

            for role in self.roles:
                try:
                    llm = random.choice(self.llms_layer)
                
                    # Apply temperature based on layer strategy
                
                    setattr(llm, "temperature", random.uniform(0.1, 1))
         
                    responses.append(await self.generate_based_on_roles(llm, user_prompt, role['name'], role['description']))
                except Exception as e:
                    print(f"Error with generating response for role {role}: {e}")
        
        elif self.layer_strategy == 'temperature':
            for i in range(self.nb_models_by_layer):
                llm = random.choice(self.llms_layer)
                # Use fixed temperature progression based on layer
                temp_step = (1.0 - 0.1) / (self.layer + 1)
                setattr(llm, "temperature", 0.1 + (temp_step * self.layer))
                responses.append(await self.generate(llm, user_prompt))
        
        elif self.layer_strategy == 'model':
            for i in range(len(self.models_layer)):
                responses.append(await self.generate(self.llms_layer[i], user_prompt))

        if len(responses) == 0:
            raise Exception("No responses generated")
        print(f"Len Responses: {len(responses)}")
        return responses
    
    def get_aggregation_prompt(self, system_prompt: str, responses: List[str], user_prompt: str):
        if isinstance(responses[0], AIMessage):
            prompt = system_prompt + "\n" + "\n".join([r.content for r in responses])
        else:
            prompt = system_prompt + "\n" + "\n".join(responses)
            
        return ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("user", user_prompt)
        ]).format_messages()

    async def aggregate_responses(self, responses: List[str], user_prompt: str, layer: int = None):
        try:
            aggregator_system_prompt = """You have been provided with a set of responses from various models to the latest user query.
            Your task is to synthesize these responses into a single, high-quality response.
            It is crucial to critically evaluate the information provided in these responses,
            recognizing that some of it may be biased or incorrect.
            Your response should not simply replicate the given answers but should offer a refined,
            accurate, and comprehensive reply to the instruction.
            Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
            Write your answer without mentioning the models or the roles.
            
            Responses from models:"""
            
            aggregation_prompt = self.get_aggregation_prompt(aggregator_system_prompt, responses, user_prompt)
            aggregation_response = await self.llm_aggregator.ainvoke(aggregation_prompt)
            await self.log_trace(
                "aggregation",
                aggregation_response.content,
                layer=layer,
                model_name=self.aggregator_model
            )
            return aggregation_response.content
        except Exception as e:
            print(e)
            from time import sleep
            sleep(5)
            return await self.aggregate_responses(responses, user_prompt, layer)

    async def run(self, user_prompt: str):
        # Initialize base entry at the start of run
        self.base_entry = {
            "user_prompt": user_prompt,
        }
        
        if self.layer_strategy == 'role':
            logging.info("Generating roles")
                
            roles = await self.generate_roles(user_prompt, self.nb_models_by_layer)
            assert len(roles['roles']) > 0, f"Expected {self.nb_models_by_layer} roles, got {len(roles['roles'])}"
            logging.info(f"Roles: {roles}")
            self.roles = roles['roles']

        

        responses = await self.generate_layer(user_prompt)
        logging.info(f"Response of each role: {responses}")
        
        
        logging.info("Aggregating responses")
        for i in range(self.layer):
            logging.info(f"Aggregating responses layer {i} on {self.layer}")
            aggregation_response = await self.aggregate_responses(responses, user_prompt, layer=i)
            
            prompt = f"The user prompt is {user_prompt}. For now the aggregated response is {aggregation_response}"
            responses = await self.generate_layer(prompt)
        
        logging.info("Final aggregation")
        final_response = await self.aggregate_responses(responses, user_prompt, layer="final")
        
        return final_response

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--user_prompt", type=str, default="""Problème : La Société des Couleurs et des Formes

Dans une société particulière, les gens sont divisés en deux groupes :
- Les Cercles : Ils disent toujours la vérité sur les couleurs mais mentent sur les formes
- Les Carrés : Ils disent toujours la vérité sur les formes mais mentent sur les couleurs

Vous rencontrez trois résidents : Alice, Bob et Charlie.

Alice dit : "Je vois un triangle rouge."
Bob dit : "Ce même objet est un triangle bleu."
Charlie dit : "L'objet dont parlent Alice et Bob est un carré rouge."

Questions :
1. Quelle est la véritable couleur et forme de l'objet ?
2. Quel type (Cercle ou Carré) est chaque résident ?""")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--models-layer", type=str, nargs='+', default=["gpt-4o-mini", 'claude-3-5-haiku-latest'], help="List of models to use for the layer for delimiting in args use --layer-strategy model1,model2,model3")
    parser.add_argument("--nb-models-by-layer", type=int, default=2)
    parser.add_argument("--model-roles", type=str, default="chatgpt-4o-latest")
    parser.add_argument("--aggregator-model", type=str, default="claude-3-5-haiku-latest")
    parser.add_argument("--layer-strategy", type=str, default='role', 
                       choices=['role', 'temperature', 'model'])
    args = parser.parse_args()
    
    moa = MoA(
        models_layer=args.models_layer,
        model_roles=args.model_roles,
        aggregator_model=args.aggregator_model,
        nb_models_by_layer=args.nb_models_by_layer,
        layer=args.layer,
        layer_strategy=args.layer_strategy
    )

    result = asyncio.run(moa.run(args.user_prompt))
    print(result)