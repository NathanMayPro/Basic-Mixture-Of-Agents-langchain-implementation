from typing import Any, Dict, List, Optional, Iterator
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import SecretStr, Field
from moa import MoA



class MoALLM(LLM):
    """A custom LLM that implements Mixture of Agents (MoA) architecture.
    
    This LLM uses multiple models in parallel to generate responses from different roles/perspectives,
    then aggregates them into a final response.

    Example:
        .. code-block:: python

            llm = MoALLM(
                models_layer=["mistral-small", "mistral-medium"],
                model_roles="mistral-medium",
                aggregator_model="mistral-medium-latest",
                layer_depth=2
            )
            result = llm.invoke("What is the meaning of life?")
    """

    models_layer: List[str]
    model_roles: str
    aggregator_model: str
    layer_depth: int = 2
    layer: int = 0
    temperature: float = 0.1
    max_tokens: int = 4096
    api_key: Optional[SecretStr] = None
    moa: MoA = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the base MoA class
        self.moa = MoA(
            models_layer=self.models_layer,
            model_roles=self.model_roles,
            aggregator_model=self.aggregator_model,
            layer_depth=self.layer_depth,
            layer=self.layer,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the MoA on the given prompt.

        Args:
            prompt: The input prompt to process
            stop: Stop sequences (not supported in MoA)
            run_manager: Callback manager
            **kwargs: Additional arguments passed to the underlying MoA

        Returns:
            The aggregated response from multiple agents
        """
        import asyncio
        
        if stop is not None:
            raise ValueError("stop sequences are not supported in MoA")

        # Run the MoA asynchronously
        response = asyncio.run(self.moa.run(prompt))
        
        if run_manager:
            run_manager.on_llm_new_token(response)
            
        return response

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Async stream the MoA response.

        Args:
            prompt: The input prompt to process
            stop: Stop sequences (not supported)
            run_manager: Callback manager
            **kwargs: Additional arguments

        Yields:
            GenerationChunk objects containing response tokens
        """
        response = await self.moa.run(prompt)
        
        # Simulate streaming by yielding words
        for word in response.split():
            chunk = GenerationChunk(text=word + " ")
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get parameters that identify this LLM."""
        return {
            "model_name": "MoA",
            "models_layer": self.models_layer,
            "model_roles": self.model_roles,
            "aggregator_model": self.aggregator_model,
            "layer_depth": self.layer_depth,
            "layer": self.layer
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of LLM."""
        return "moa"


if __name__ == "__main__":
    llm = MoALLM(
        models_layer=["mistral-small", "mistral-medium"],
        model_roles="mistral-medium",
        aggregator_model="mistral-medium-latest",
        layer_depth=2,
        layer=1
    )
    
    # basic chain 
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    
    # Create a template with an input variable
    prompt = PromptTemplate.from_template("Tell me about {topic}")
    chain = prompt | llm
    
    # Provide the input as a dictionary with the template variable
    print(chain.invoke({"topic": "the meaning of life"}))
    