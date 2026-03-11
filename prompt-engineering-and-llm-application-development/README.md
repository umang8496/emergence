# Prompt Engineering and LLM Application Development

## Table of Content

- [Chapter 1: Foundations of Prompt Engineering](#chapter-1-foundations-of-prompt-engineering)
  - [Introduction to Large Language Models](#introduction-to-large-language-models)
  - [What is Prompt Engineering?](#what-is-prompt-engineering)
  - [Components of a Prompt](#components-of-a-prompt)
  - [Basic Prompting Techniques](#basic-prompting-techniques)
  - [Understanding LLM Temperature and Other Parameters](#understanding-llm-temperature-and-other-parameters)
- [Chapter 2: Advanced Prompting Strategies](#chapter-2-advanced-prompting-strategies)
  - [Zero-Shot Prompting](#zero-shot-prompting)
  - [Few-Shot Prompting](#few-shot-prompting)
  - [Instruction Following Prompts](#instruction-following-prompts)
  - [Role Prompting](#role-prompting)
  - [Structuring Output Formats(JSON, Markdown)](#structuring-output-formats)
  - [Chain-of-Thought Prompting](#chain-of-thought-prompting)
  - [Self-Consistency Prompting](#self-consistency-prompting)
- [Chapter 3: Prompt Design, Iteration, and Evaluation](#chapter-3-prompt-design-iteration-and-evaluation)
- [Chapter 4: Interacting with LLM APIs](#chapter-4-interacting-with-llm-apis)
- [Chapter 5: Building Applications with LLM Frameworks](#chapter-5-building-applications-with-llm-frameworks)
- [Chapter 6: Integrating LLMs with External Data (RAG)](#chapter-6-integrating-llms-with-external-data-rag)
- [Chapter 7: Output Parsing, Validation, and Application Reliability](#chapter-7-output-parsing-validation-and-application-reliability)
- [Chapter 8: Application Development Considerations](#chapter-8-application-development-considerations)
- [References](#references)

<br /><!-- markdownlint-disable-line MD033 -->

## Chapter 1: Foundations of Prompt Engineering

### Introduction to Large Language Models

**Large Language Models** (LLMs) represent a major advancement in artificial intelligence, particularly within natural language processing (NLP).  
They are deep learning systems trained on extremely large volumes of textual data, often spanning terabytes collected from books, websites, research papers, and code repositories.  

The term **large** refers to two key aspects: the massive size of the training data and the enormous number of parameters—ranging from billions to trillions — that the model learns during training.  
These parameters enable the model to internalize grammatical patterns, semantic relationships, contextual dependencies, and even reasoning-like structures found in human language.  

Most modern LLMs are built on the Transformer architecture, introduced in the landmark paper **Attention Is All You Need**.  
The core innovation of this architecture is the attention mechanism, which allows the model to determine the relative importance of words in a sequence when generating output.  
This design enables LLMs to effectively handle long-range dependencies in text and maintain contextual coherence across extended passages.  

At a fundamental level, an LLM functions as an advanced text prediction engine.  
Given an input prompt, it predicts the next most likely token (word or sub-word unit), and then continues this prediction process iteratively to generate complete and contextually relevant responses.  
Despite this seemingly simple mechanism, the scale and training of these models allow them to perform a wide range of sophisticated language tasks.  

From an application development perspective, this predictive capability translates into several practical use cases:

- **Text Generation** – Creating original content such as emails, blog posts, marketing copy, stories, scripts, or code snippets.
- **Summarization** – Condensing lengthy documents into concise summaries while preserving key ideas.
- **Question Answering** – Responding to user queries based on embedded knowledge or contextual information provided in the prompt.
- **Translation** – Converting text between different languages.
- **Classification** – Categorizing text, such as performing sentiment analysis or topic labeling.
- **Information Extraction** – Identifying and extracting structured data (e.g., names, dates, entities) from unstructured text.
- **Code Generation** – Producing executable code from natural language instructions.

The interaction model is straightforward: a user provides a textual input (prompt), and the LLM generates a corresponding textual output.  
This simplicity, combined with broad linguistic capability, makes LLMs a foundational technology for building intelligent, language-driven applications.

---

### What is Prompt Engineering?

**Prompt engineering** is the disciplined practice of designing and refining the input text—known as a **prompt** — to guide an LLM toward producing a specific, reliable, and well-structured output.  
While LLMs are highly capable of generating human-like text, code, and other forms of content, simply providing access to a model does not guarantee useful or consistent results.  
The quality of the output depends significantly on how effectively the task is communicated.  
In this sense, prompt engineering resembles giving precise instructions to a highly knowledgeable but probabilistic collaborator rather than writing deterministic program logic.  

Unlike traditional software systems, which execute clearly defined and rule-based instructions, LLMs operate probabilistically.  
In conventional programming, logic is explicit and outcomes are predictable for known inputs.  

For example, a function written to return the capital of France will always produce “Paris” when the condition matches.  
In contrast, when interacting with an LLM, a user provides natural language input, and the model predicts the most likely continuation based on patterns learned during training.  

Although the model may often provide the correct answer, the output is not strictly guaranteed, and small variations in phrasing can influence the response.  
As tasks become more complex—such as enforcing strict formatting rules or extracting structured data—the importance of precise prompt construction increases substantially.  

Poorly constructed prompts can lead to several issues, including:

- Vague or irrelevant responses
- Outputs in an unusable or incorrect format
- Factually incorrect information (hallucinations)
- Failure to follow specific instructions

Conversely, a well-engineered prompt functions as a clear specification. It reduces ambiguity and aligns the model's probabilistic behavior with the user's intent.  

Effective prompt engineering combines structured communication, an understanding of model behavior, and iterative refinement. Key components typically include:

- **Instruction Clarity** – Explicitly stating the task the model must perform.
- **Context Provision** – Supplying relevant background information needed for accurate output.
- **Input Data** – Clearly defining the specific content the model should operate on.
- **Output Formatting** – Specifying how the response should be structured (e.g., JSON, bullet points, code).

The interaction flow generally follows this sequence: a user's need is translated into an engineered prompt that integrates instructions, context, and data;  
The LLM processes this prompt; and the model produces the desired output in the requested format.  

Because LLM responses can vary depending on phrasing and context, prompt engineering is rarely a one-step process.  
Instead, it follows an iterative cycle:

1. **Design** – Create an initial prompt aligned with task requirements.
2. **Test** – Submit the prompt to the LLM and observe the output.
3. **Analyze** – Evaluate whether the output meets accuracy, formatting, and completeness criteria.
4. **Refine** – Adjust the prompt to address gaps or inconsistencies, then repeat the cycle.

This iterative refinement is central to building reliable LLM-powered systems.  
Mastery of prompt engineering enables developers to move from basic task instructions to structured, dependable, and scalable application behavior.

---

### Components of a Prompt

Prompts are not merely questions posed to a Large Language Model (LLM); they function as structured task specifications.  
Effective prompt engineering requires understanding the internal structure of a prompt and the role each element plays in guiding model behavior.  
Although prompts can range from extremely simple to highly detailed, most well-constructed prompts consist of four fundamental components working together to produce reliable and controlled outputs.

These core components are:

- **Instruction** – Defines the task the LLM should perform.
- **Context** – Supplies background information, constraints, or domain-specific guidance.
- **Input Data** – Provides the specific information the model must process.
- **Output Indicator** – Specifies the desired format or structure of the response.

Not every prompt requires all four elements.  
Simple requests may only include an instruction and input data.  
However, for complex or production-grade applications, explicitly defining each component significantly improves clarity, predictability, and output quality.  

#### Instruction

The instruction is the operational directive of the prompt. It clearly states what action the model must take. Precision in wording directly affects output reliability.

Examples include:

- “Summarize the following text.”
- “Translate the following English sentence into French.”
- “Write Python code to read a CSV file and print the first five rows.”

The instruction establishes the primary objective of the generation process.  
Vague instructions often lead to ambiguous or incomplete outputs, whereas specific task-oriented instructions constrain the model toward the intended behavior.  

#### Context

Context provides supplementary information, constraints, or assumptions that shape how the model executes the instruction. It may include:

- Project-specific details not inherently available to the model.
- Output constraints (e.g., length limits, tone requirements, persona assumptions).
- Domain knowledge or classification categories.

For example, if drafting a response about a newly launched product, the context might describe the product's features and available variations.  
This background enables the model to generate an accurate and informed answer.  

In structured prompts, context acts as environmental conditioning — it narrows the interpretation space and reduces unintended responses.  

#### Input Data

Input data is the concrete material the model must operate on. It is the subject of transformation according to the instruction and within the provided context.  

Depending on the task, input data may include:

- A long article (for summarization)
- A sentence in a source language (for translation)
- A customer review (for sentiment classification)
- A functional requirement description (for code generation)

Clear separation of input data within a prompt improves focus and reduces the risk of the model misinterpreting instructions as data.

#### Output Indicator

The output indicator defines how the response should be structured.  
It ensures that the generated output is directly usable within an application workflow.  
This can range from subtle cues to strict format enforcement.  

Examples include:

- Simple markers such as “Summary:” or “Translation:”
- Explicit formatting requirements like “Respond in JSON with keys ‘original_sentence' and ‘translated_sentence'.”
- Structured outputs such as “Return the result as a Python list of strings.”

The output indicator is especially critical when integrating LLM responses into automated systems, APIs, or pipelines where format consistency is essential.  

#### Combining the Components

When these elements are combined systematically, they form a robust prompt specification. For example, in a sentiment classification task:

- The **Instruction** defines the classification task.
- The **Context** specifies allowable sentiment categories and response constraints.
- The **Input Data** provides the customer review text.
- The **Output Indicator** signals the expected format (e.g., returning only the category name).

This structured composition reduces ambiguity and improves response consistency.  

Prompt engineering, therefore, is both analytical and iterative.  
Adjusting how each component is phrased or ordered can materially influence model output.  
Understanding these building blocks enables systematic experimentation, more reliable behavior, and ultimately more effective LLM-powered applications.  

---

### Basic Prompting Techniques

Basic prompting techniques form the foundation of interacting effectively with Large Language Models (LLMs).  
While advanced prompting strategies exist, many practical tasks can already be accomplished by simply giving the model a clear instruction.  
Understanding how prompts are structured and how parameters such as **temperature** influence response generation is essential for guiding the model toward the desired output.  

At the most fundamental level, prompt engineering involves communicating a task clearly so that the LLM can generate the appropriate response.  
Because LLMs are trained on massive datasets and have learned patterns across language, they can often perform tasks without needing explicit examples in the prompt.  
This ability enables a technique known as **zero-shot prompting**, where the model is given a direct instruction and asked to complete the task based solely on its prior training.  

#### Direct Instruction (Zero-Shot Prompting)

The simplest prompting approach is to provide a direct instruction describing the task to be performed.  
The model interprets the instruction and generates a response accordingly.  
This technique is widely used for tasks such as summarization, translation, and answering factual questions.  

Common applications include:

- **Text Summarization** – Condensing a long passage into a shorter summary while preserving key ideas.
- **Question Answering** – Asking general knowledge questions that the model answers using its learned information.
- **Translation** – Converting text from one language into another.

For example, a prompt such as **“Summarize the following text into two sentences”** clearly defines the task, enabling the model to generate a concise summary.  
Similarly, a prompt like **“What is the capital of France?”** allows the model to respond with factual knowledge it has learned during training.  

#### Text Generation and Continuation

LLMs are also capable of generating creative or structured content. By providing a starting prompt or theme, the model can produce poems, stories, code, or other forms of text.  

Typical uses include:

- **Creative Writing** – Generating poems, short stories, or descriptive passages.
- **Story Continuation** – Extending a partially written narrative.
- **Content Generation** – Producing articles, scripts, or descriptive text based on a topic.

For instance, prompting the model with **“Write a short poem about a rainy day in the city”** encourages creative text generation,  
while starting a story and asking the model to continue it allows the model to extend the narrative coherently.  

#### Information Extraction

Another common prompting technique involves extracting specific pieces of information from unstructured text.  
Instead of generating new content, the model identifies and returns particular details requested in the prompt.  

Examples of extracted information may include:

- Names of people or organizations
- Dates or locations
- Key entities or attributes within a sentence or paragraph

A prompt such as **“Extract the name of the person and the company mentioned in the following sentence”** directs the model to identify structured data from the given text.  

#### Simple Classification

LLMs can also categorize text according to predefined labels.  
This technique is useful for tasks such as sentiment analysis or topic classification.  

Examples of classification tasks include:

- **Sentiment Analysis** – Determining whether a review is positive, negative, or neutral.
- **Topic Classification** – Assigning a category to an article or message.
- **Content Labeling** – Identifying the type or intent of a piece of text.

For example, a prompt may ask the model to classify a customer review into **Positive**, **Negative**, or **Neutral** sentiment based on the tone of the message.

#### Role of Prompt Clarity and Generation Parameters

Although these prompting techniques are relatively simple, the quality of the results depends greatly on the clarity of the instruction and the generation parameters used.  
Parameters such as **temperature** influence how deterministic or creative the model's output will be.  

- **Lower temperature values** tend to produce more precise and predictable responses, making them suitable for tasks like summarization, extraction, or factual answering.
- **Higher temperature values** introduce more randomness, which can be beneficial for creative tasks such as storytelling or poetry generation.

These basic prompting methods illustrate the core principle of prompt engineering: carefully structuring input to guide the model's behavior.  
While simple, they are powerful enough to support many real-world applications.  
More advanced prompting strategies build upon these fundamentals to achieve greater control, reliability, and sophistication in LLM-powered systems.

---

### Understanding LLM Temperature and Other Parameters

Large Language Models (LLMs) generate text by predicting the next token (a word or part of a word) based on probability distributions learned during training.  
When a prompt is provided, the model calculates the probability of many possible next tokens and then selects one of them.  

**Generation parameters** act as control mechanisms that influence how the model makes this selection.  
By adjusting these parameters, developers can control aspects such as `randomness`, `creativity`, and `response length`, allowing the model to behave `more predictably` or `more creatively` depending on the task.

In practical applications, these parameters function like tuning knobs for the generation process.  
Tasks such as factual question answering or code generation typically require precise and deterministic outputs,  
whereas creative writing or brainstorming may benefit from more diverse and exploratory responses.  
Understanding how these parameters influence token selection helps developers obtain consistent and task-appropriate results from LLMs.  

#### Temperature

**Temperature** controls the randomness of the model's output. It adjusts how strongly the model prefers the most probable tokens when generating the next piece of text.

- **Low Temperature (≈ 0.1 – 0.4)**
  - Produces deterministic and focused outputs
  - The model consistently selects the most probable tokens
  - Useful for tasks requiring accuracy and reliability

  Typical use cases:

  - Factual question answering
  - Code generation
  - Text summarization
  - Classification

- **High Temperature (≈ 0.8 – 1.5)**

  - Introduces randomness and diversity in token selection
  - Allows less probable tokens to be chosen
  - Produces more creative and varied outputs

  Typical use cases:

  - Brainstorming ideas
  - Creative writing (stories, poems)
  - Generating multiple alternative responses
  - Conversational chatbots

> Conceptually, lower temperatures **sharpen the probability distribution**, making the most likely token dominant,  
> while higher temperatures **flatten the distribution**, increasing the likelihood of selecting less probable tokens.  

#### Top-p (Nucleus Sampling)

**Top-p**, also known as **nucleus sampling**, limits the set of candidate tokens based on cumulative probability rather than a fixed count.  

Instead of considering all possible tokens, the model selects the **smallest group of tokens whose combined probability exceeds a threshold (p)**.  
The next token is then sampled only from this subset.  

Example:

- If **top_p = 0.9**, the model selects tokens in descending probability order until their cumulative probability reaches **90%**.
- Only those tokens are considered for sampling.

Advantages of top-p sampling:

- Avoids selecting extremely low-probability tokens
- Dynamically adapts to the probability distribution
- Maintains diversity while preventing unrealistic outputs

Because it adjusts automatically to the distribution shape, many practitioners prefer **top-p over top-k** for balancing creativity and quality.

#### Top-k Sampling

**Top-k** sampling restricts the model to choosing from only the **k most probable tokens**.  

Example:

- If **top_k = 5**, the model identifies the five most likely tokens for the next position.
- All other tokens are ignored.
- The next token is sampled only from these five options.

Characteristics of top-k:

- Simple and easy to understand
- Provides strict control over candidate tokens
- Can sometimes be too restrictive if good tokens fall outside the top-k set

Because of this limitation, top-p is often preferred since it dynamically adjusts the token pool instead of using a fixed cutoff.

#### Other Important Generation Controls

Although temperature, top-p, and top-k control randomness, additional parameters influence other aspects of generation:

- **Max Tokens (or Max Length)** – Limits the maximum number of tokens the model can generate, preventing overly long responses and helping manage API costs.
- **Frequency Penalty** – Reduces the likelihood of repeating tokens that have already appeared frequently in the output.
- **Presence Penalty** – Encourages the introduction of new topics or tokens by penalizing those that have appeared before.
- **Stop Sequences** – Defines specific strings that terminate generation once produced.

#### Choosing the Right Parameter Settings

There is no universal parameter configuration suitable for every task. The optimal settings depend on the desired behavior of the model.

Typical starting points include:

- **Factual or deterministic tasks**

  - Low temperature (0.1–0.4)
  - top_p close to 1.0

- **Creative or exploratory tasks**

  - Higher temperature (0.7–1.0)
  - top_p between 0.8 and 0.95

- **Balanced tasks**

  - Moderate temperature (0.5–0.7)

In practice, developers often refine these values through experimentation.  
By adjusting parameters incrementally and observing the results, it becomes possible to achieve the desired balance between creativity, precision, and response length.  

Understanding these generation parameters provides deeper control over how an LLM produces text.  
Combined with well-structured prompts, they form a powerful toolkit for building reliable and effective AI-driven applications.  

---

#### [Chapter 01: Hands On Exercise](./notebooks/chapter-01-hands-on.ipynb)

---

## Chapter 2: Advanced Prompting Strategies

### Zero-Shot Prompting

Zero-shot prompting is a fundamental prompting strategy used when working with Large Language Models (LLMs).  
In this approach, the model is asked to perform a task using only a clear instruction or description, without providing any examples of the task within the prompt.  
Instead of demonstrating how the task should be performed, the user simply states what needs to be done and relies on the model's extensive training to understand and execute the request.  

This method works effectively because modern LLMs are trained on massive datasets containing diverse forms of text such as books, websites, articles, and code repositories.  
During training, the models learn patterns associated with common language tasks including summarization, translation, question answering, classification, and information extraction.  
As a result, when a task description resembles patterns encountered during training, the model can often generalize and complete the task successfully without needing explicit examples.  

#### How Zero-Shot Prompting Works

The effectiveness of zero-shot prompting lies in the model's ability to apply previously learned knowledge to new instructions.  
When a user provides a prompt describing a task, the model interprets the instruction and generates a response based on patterns learned during training.  
Instruction-tuned models are especially good at following such prompts because they have been optimized to understand and respond to natural language instructions.  

In practice, a zero-shot prompt typically consists of a clear instruction followed by the input data on which the task should be performed.  

Examples of Zero-Shot Prompting

Zero-shot prompting can be applied to a wide range of tasks. Some common examples include:

##### Sentiment Classification

A model can be asked to determine the sentiment of a piece of text without seeing any example classifications.

```text
Classify the sentiment of the following customer review as positive, negative, or neutral.

Review: "The setup process was incredibly confusing, but the support team was very helpful."

Sentiment:
```

##### Text Summarization

The model can summarize a long piece of content based solely on the instruction.

```text
Summarize the main points of the following article in three bullet points.
```

##### Extracting Information

Specific entities can be extracted from text, often with a requested output format.

```text
Extract the name of the person and the company from the following sentence and return the result as JSON.
```

##### Language Translation

Translation tasks are particularly well-suited for zero-shot prompting.

```text
Translate the following English text to French:

"Hello, how are you?"
```

#### Advantages of Zero-Shot Prompting

Zero-shot prompting offers several practical benefits:

- **Simplicity** – It is the most straightforward prompting approach since it only requires a clear instruction.
- **No Example Data Required** – There is no need to collect or prepare example inputs and outputs.
- **Quick Baseline Evaluation** – It allows developers to quickly test whether an LLM can perform a task effectively without additional prompt engineering.

#### Limitations and Challenges

Despite its usefulness, zero-shot prompting may not always produce optimal results. Some common challenges include:

- **Performance Limitations** – For highly specialized or complex tasks, the model may perform better when examples are provided.
- **Dependence on Instruction Clarity** – Ambiguous or poorly written prompts can lead to inconsistent or incorrect responses.
- **Model Capability Differences** – Some models handle zero-shot tasks better than others, particularly larger or instruction-tuned models.

#### When to Use Zero-Shot Prompting

Zero-shot prompting is usually the best starting point when developing applications that involve LLMs.  
For common NLP tasks such as classification, translation, summarization, or information extraction, a well-written zero-shot prompt may be sufficient to achieve good results.  

If the model's performance is not satisfactory,  
the zero-shot prompt can serve as a baseline before moving to more advanced techniques such as few-shot prompting, where examples are included to guide the model's behavior.  

Understanding how to design effective zero-shot prompts is an essential skill in prompt engineering, as it leverages the inherent capabilities of modern language models with minimal effort.

---

### Few-Shot Prompting

Few-shot prompting is an advanced prompt engineering technique used to guide Large Language Models (LLMs) by providing a small number of examples directly within the prompt.  
Unlike zero-shot prompting, where the model receives only an instruction, few-shot prompting demonstrates how the task should be performed by including example input–output pairs.  
The model analyzes these examples and learns the pattern during the current interaction, enabling it to apply the same pattern to new input.  

This technique relies on a mechanism called **in-context learning**, where the model infers the relationship between inputs and outputs from examples provided in the prompt.  
Importantly, the model's internal parameters are not modified.  
The learning happens temporarily within the context of the prompt, allowing the model to adapt its behavior dynamically for the given task.  

Few-shot prompting is particularly useful when:

- The task is **novel or domain-specific** and cannot be easily described with a simple instruction.
- The output must follow a **specific format**.
- You want the model to adopt a **particular style, tone, or reasoning pattern**.
- The results from zero-shot prompting are not sufficiently accurate or consistent.

#### How Few-Shot Prompting Works

When examples are provided, the model identifies the pattern connecting each input with its corresponding output.  
It then applies that same pattern to the final query. The examples act as demonstrations rather than training data.  

You can think of this process as giving a colleague quick instructions with examples right before they perform a task.  
The colleague observes the examples, understands the pattern, and applies it to the new problem.  

#### Structure of a Few-Shot Prompt

A typical few-shot prompt follows a structured format:

1. **Optional Task Description** – Briefly explains the task.
2. **Example Inputs and Outputs** – Demonstrates how the task should be performed.
3. **Actual Input** – The query the model must solve.
4. **Output Indicator** – Signals where the model should generate the answer.

Consistency in formatting and labeling across examples is essential for good results.  

##### Example 1: Sentiment Classification

Few-shot prompting can help guide classification tasks by showing examples of how inputs should be labeled.  

```text
Classify the sentiment of the following customer reviews.

Text: "The battery life on this device is amazing!"
Sentiment: Positive

Text: "The screen scratches too easily."
Sentiment: Negative

Text: "It arrived on time."
Sentiment: Neutral

Text: "Customer support was helpful but slow to respond."
Sentiment:
```

In this prompt, the examples demonstrate:

- The task (sentiment classification)
- The expected labels (Positive, Negative, Neutral)
- The formatting of the output

The model then predicts the sentiment for the final review based on the demonstrated pattern.

##### Example 2: Code Generation

Few-shot prompting can also guide code generation tasks.  
By providing examples of how descriptions translate into functions, the model learns the structure and logic required.  

```python
# Generate a Python function based on the description.

Description: """Adds two numbers."""
Function:
def add(a, b):
    """Adds two numbers."""
    return a + b

Description: """Concatenates two strings with a space."""
Function:
def concatenate_strings(s1, s2):
    """Concatenates two strings with a space."""
    return s1 + " " + s2

Description: """Calculates the area of a rectangle."""
Function:
```

The model observes that:

- The description becomes the function's docstring.
- The function name reflects the description.
- Appropriate parameters and logic are implemented.

Based on this pattern, the model will likely generate a function that calculates the area of a rectangle.

##### Choosing Effective Examples

The success of few-shot prompting depends heavily on the examples included in the prompt. Some important guidelines include:

- **Relevance** – Examples should closely match the task and the type of input expected.
- **Consistency** – Use the same structure and labeling for all examples and the final query.
- **Accuracy** – Incorrect examples will mislead the model.
- **Variety** – If the task involves different cases (e.g., multiple sentiment categories), include examples representing those cases.
- **Conciseness** – Examples should be short enough to conserve the context window.

Typically, **one to five examples** are sufficient for effective few-shot prompting.

#### Trade-offs and Limitations

While few-shot prompting provides more control than zero-shot prompting, it also introduces several trade-offs:

- **Context Window Usage** – Including examples increases the number of tokens in the prompt.
- **Example Sensitivity** – Model performance may depend strongly on the specific examples used or their order.
- **Higher Cost** – Longer prompts increase API usage costs.
- **Temporary Learning** – The model does not permanently learn from examples; the guidance only applies within that prompt.

For tasks requiring consistent, large-scale performance, **fine-tuning the model** may be a more appropriate long-term solution.  

Few-shot prompting is a powerful technique that improves LLM performance by demonstrating how a task should be performed through examples.  
By leveraging **in-context learning**, it allows developers to guide model behavior without modifying the model itself.  
Compared to zero-shot prompting, few-shot prompting provides greater control over output format, reasoning style, and task execution, making it a valuable tool for building reliable AI-driven applications.  

---

### Instruction Following Prompts

Instruction following prompts are a prompt engineering technique used to guide Large Language Models (LLMs) through **explicit and detailed instructions**.  
Instead of asking vague questions, this approach provides clear commands that define exactly what the model should do, how it should perform the task, and how the output should be structured.  
This technique helps achieve more **reliable, consistent, and controlled outputs**, which is particularly important when building real-world applications.  

Unlike few-shot prompting, which relies on examples to demonstrate a pattern, instruction following focuses primarily on **clear directives and constraints**.  
The goal is to reduce ambiguity so that the model can execute the task precisely according to the provided instructions.  

#### Key Components of Instruction Following Prompts

Effective instruction prompts typically include several elements that clarify the task and the expected output.

##### 1. Clear Action Verb

The prompt should begin with a strong action verb that defines the task. Common verbs include:

- Summarize
- Translate
- Extract
- Generate
- Rewrite
- Classify
- Compare
- Explain

These verbs clearly signal the type of operation the model must perform.

##### 2. Subject or Input Specification

The prompt must clearly define the **content or input*- the model should process. This might include:

- A block of text
- An email message
- A dataset
- Code or documentation

Providing the input directly in the prompt ensures the model understands what it needs to analyze.

##### 3. Constraints and Requirements

Constraints define **rules or boundaries** for the output. These help narrow the response and make it more predictable.  

Examples include:

- **Length constraints:** “in under 100 words”, “exactly three bullet points”
- **Content constraints:** “focus on the technical aspects”
- **Tone constraints:** “write in a formal tone”
- **Audience constraints:** “explain for a beginner”

##### 4. Output Format Specification

Explicitly defining the structure of the output ensures the result can be easily processed or integrated into applications.  

Examples include:

- JSON objects
- Markdown tables
- Bullet lists
- Comma-separated values

For example, a prompt may ask the model to return information as:

```json
{
  "name": "",
  "summary": ""
}
```

##### 5. Negative Constraints (Optional)

Sometimes it is helpful to specify what the model should **avoid doing**. This can prevent unwanted behavior.

Examples include:

- “Do not include personal opinions.”
- “Avoid overly technical language.”
- “Do not add an introduction or conclusion.”

#### Examples: Improving Prompt Precision

##### Example 1: Summarization

Vague Prompt

```text
Summarize this text.
```

Instruction-following Prompt

```text
Summarize the following text in exactly two sentences, focusing on the author's main conclusion. Do not include examples from the text.
```

This version clearly defines both **length** and **content constraints**.

##### Example 2: Information Extraction

Vague Prompt

```text
Find important information in this email.
```

Instruction-following Prompt

```text
Extract the sender's name, meeting date, and meeting time from the email. 
Return the output as a JSON object with keys:
"sender_name", "meeting_date", and "meeting_time".
If any information is missing, return null.
```

Here the instructions specify **what to extract** and **how to format the output**.

##### Example 3: Code Generation

Vague Prompt

```text
Write Python code to read a file.
```

Instruction-following Prompt

```text
Generate a Python function named read_text_file that takes file_path as input.
The function should:
1. Open the file in read mode.
2. Read the entire content.
3. Handle FileNotFoundError by returning None.
4. Return the file content as a string.

Include a docstring explaining the function and its arguments.
Do not include example usage.
```

This prompt clearly specifies **function name, parameters, behavior, error handling, and documentation**.

#### Best Practices for Writing Instructions

To design effective instruction-following prompts:

- **Be explicit and unambiguous.** Avoid vague wording.
- **Use action-oriented language** to clearly define the task.
- **Structure complex instructions** with numbered steps or bullet points.
- **Keep instructions simple and readable.**
- **Iterate and refine prompts** based on the model’s responses.

Prompt engineering often involves experimenting and adjusting instructions until the desired behavior is achieved.

#### When to Use Instruction Following Prompts

Instruction-following prompts are especially useful when:

- The task involves **multiple steps or complex logic**.
- A **specific output format** is required.
- The response must follow a **specific tone or style**.
- The output will be used in **automated pipelines or applications**.

While zero-shot and few-shot prompting can handle many tasks, instruction following provides **greater control and precision**, making it an essential technique for developing reliable LLM-powered systems.

---

### Role Prompting

---

### Structuring Output Formats

---

### Chain-of-Thought Prompting

---

### Self-Consistency Prompting

---

#### [Chapter 02: Hands On Exercise](./notebooks/chapter-02-hands-on.ipynb)

---

## Chapter 3: Prompt Design, Iteration, and Evaluation

## Chapter 4: Interacting with LLM APIs

## Chapter 5: Building Applications with LLM Frameworks

## Chapter 6: Integrating LLMs with External Data (RAG)

## Chapter 7: Output Parsing, Validation, and Application Reliability

## Chapter 8: Application Development Considerations

<br /><!-- markdownlint-disable-line MD033 -->

## References

- [Prompt Engineering and LLM Application Development Course](https://apxml.com/courses/prompt-engineering-llm-application-development)

<br /><!-- markdownlint-disable-line MD033 -->
