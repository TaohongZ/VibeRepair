from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import re


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class CorrectSpecGeneratorMF:
    
    PRICE_INPUT = 5.0 / 1_000_000
    PRICE_OUTPUT = 15.0 / 1_000_000
    
    def __init__(self, model_name: str = "gpt-4o-2024-05-13", temperature: float = 1.0):
        self.model = ChatOpenAI(model=model_name, temperature=temperature)
        self.system_message_text = (
            "You are an expert Java programmer and specification/code analyzer. "
            "The user will provide you with several flawed specifications, along with the correct intended behavior, root causes, and suggestions for fixing those flawed specifications."
            "Based on all the provided information, you must revise all the flawed specifications and produce the corresponding correct specifications."
            "Your output must strictly present each final correct specification in order."
            "Your output must contain only the correct specifications and nothing else."
        )
        
        self.messages = []
    
    def _calc_cost(self, response) -> float:
        
        usage = response.usage_metadata
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]

        return (
            input_tokens * self.PRICE_INPUT
            + output_tokens * self.PRICE_OUTPUT
        )

    def start_session(self, flawed_speci: str, cib: str, root_cause: str, suggestion: str) -> tuple[str, float]:
        

        
        system_msg = SystemMessage(content=self.system_message_text)

        human_content = (
            f"Flawed specifications:\n{flawed_speci}\n\n"
            f"Correct intended behavior: {cib}\n\n"
            f"Root cause: {root_cause}\n\n"
            f"Suggestions: {suggestion}"
        )
        human_msg = HumanMessage(content=human_content)

        
        self.messages = [system_msg, human_msg]

        
        response = self.model.invoke(self.messages)

        
        ai_msg = AIMessage(content=response.content)
        self.messages.append(ai_msg)
        
        cost = self._calc_cost(response)

        
        return response.content, cost

    def add_message(self, user_message: str) -> str:
        
       
        human_msg = HumanMessage(content=user_message)
        self.messages.append(human_msg)

        
        response = self.model.invoke(self.messages)

        
        ai_msg = AIMessage(content=response.content)
        self.messages.append(ai_msg)
        
        cost = self._calc_cost(response)

        
        return response.content, cost

    def get_history(self):
        
        return self.messages


class CorrectSpecGeneratorSF:
    
    PRICE_INPUT = 5.0 / 1_000_000
    PRICE_OUTPUT = 15.0 / 1_000_000
    
    def __init__(self, model_name: str = "gpt-4o-2024-05-13", temperature: float = 1.0):
        self.model = ChatOpenAI(model=model_name, temperature=temperature)
        self.system_message_text = (
            "You are an expert Java programmer and specification/code analyzer. "
            "The user will provide you with a flawed specification, along with the correct intended behavior, "
            "root cause, and suggestions for fixing the flawed specification. Based on all the provided information, "
            "you must revise the flawed specification and produce a correct specification.\n\n"
            "Your output must strictly contain only one final correct specification, and nothing else."
        )
        
        self.messages = []
    
    def _calc_cost(self, response) -> float:
        
        usage = response.usage_metadata
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]

        return (
            input_tokens * self.PRICE_INPUT
            + output_tokens * self.PRICE_OUTPUT
        )

    def start_session(self, flawed_speci: str, cib: str, root_cause: str, suggestion: str) -> tuple[str, float]:
        

        
        system_msg = SystemMessage(content=self.system_message_text)

        human_content = (
            f"Flawed specification:\n{flawed_speci}\n\n"
            f"Correct intended behavior: {cib}\n\n"
            f"Root cause: {root_cause}\n\n"
            f"Suggestions: {suggestion}"
        )
        human_msg = HumanMessage(content=human_content)

        
        self.messages = [system_msg, human_msg]

        
        response = self.model.invoke(self.messages)

        
        ai_msg = AIMessage(content=response.content)
        self.messages.append(ai_msg)
        
        cost = self._calc_cost(response)

        
        return response.content, cost

    def add_message(self, user_message: str) -> str:
        
        
        human_msg = HumanMessage(content=user_message)
        self.messages.append(human_msg)

        
        response = self.model.invoke(self.messages)

        
        ai_msg = AIMessage(content=response.content)
        self.messages.append(ai_msg)
        
        cost = self._calc_cost(response)

        
        return response.content, cost

    def get_history(self):
        
        return self.messages

def spec_gen_code_sf(fix_speci, buggy_code) -> str:
    
    PRICE_INPUT = 5.0 / 1_000_000
    PRICE_OUTPUT = 15.0 / 1_000_000
    
    model = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=1)

    buggy_code_text = f"Buggy code:\n{buggy_code}\n"


    
    system_message = """
    You are a code analysis expert. The user will provide you with a correct specifications and the corresponding buggy code. 
    You will fix the incorrect parts of the buggy code according to the correct specification.

    Your output should follow the format below:
    Fix code:
    '''
    #fix_code
    '''

    Example of output:
    Fix code:
    '''
    protected void burstToken(String token, boolean stopAtNonOption) {
    int tokenLength = token.length();

    for (int i = 1; i < tokenLength; i++) {
        String ch = String.valueOf(token.charAt(i));
        boolean hasOption = options.hasOption(ch);

        if (hasOption) {
            tokens.add("-" + ch);
            currentOption = options.getOption(ch);

            if (currentOption.hasArg() && (token.length() != (i + 1))) {
                tokens.add(token.substring(i + 1));
                break;
            }
        } else if (stopAtNonOption) {
            process(token.substring(i));
        } else {
            tokens.add(token);
            break;
        }
    }
}

    '''
    
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_message}"),
            (
                "human",
                "Correct specification:\n{fix_speci}\n\n{buggy_code_text}",
            ),
        ]
    )

    
    messages = prompt_template.format_messages(
        system_message=system_message,
        fix_speci=fix_speci,
        buggy_code_text=buggy_code_text,
    )

    
    response = model.invoke(messages)
    
    fix_code_text = response.content

    
    usage = response.usage_metadata
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]

    
    cost = (
        input_tokens * PRICE_INPUT
        + output_tokens * PRICE_OUTPUT
    )

    return fix_code_text, cost


def extract_fix_code(text: str) -> str:
    
    pattern = r"Fix code:\s*```[a-zA-Z]*\s*([\s\S]*?)```"

    match = re.search(pattern, text)
    if not match:
        raise ValueError("Could not find code block after 'Fix code:'")

    code = match.group(1).strip()

    if not code.endswith("\n"):
        code += "\n"

    return code

def spec_gen_code(fix_speci,buggy_codes) -> str:
    PRICE_INPUT = 5.0 / 1_000_000
    PRICE_OUTPUT = 15.0 / 1_000_000

    
    model = ChatOpenAI(model="gpt-4o", temperature=1)

    buggy_code_text = "\n".join(
        f"Buggy code {i + 1}:\n{code}\n"
        for i, code in enumerate(buggy_codes)
    )

    
    system_message = """
    You are a code analysis expert. The user will provide you with several correct specifications and the corresponding buggy code. 
    You will fix the incorrect parts of the buggy code according to the correct specification.
    
    Your output should follow the format below:
    Fix code 1:
    '''
    #fix_code
    '''
    
    Fix code 2:
    '''
    #fix_code
    '''
    
    Example of output:
    Fix code 1:
    '''
    private void handleProperties(Properties properties) throws ParseException {
    if (properties == null) {
        return;
    }

    for (Enumeration<?> e = properties.propertyNames(); e.hasMoreElements();) {
        String option = e.nextElement().toString();

        Option opt = options.getOption(option);
        if (opt == null) {
            throw new UnrecognizedOptionException(
                "Default option wasn't defined", option
            );
        }

        // if the option is part of a group, check if another option of the group has been selected
        OptionGroup group = options.getOptionGroup(opt);
        boolean selected = (group != null && group.getSelected() != null);

        if (!cmd.hasOption(option) && !selected) {

            // get the value from the properties
            String value = properties.getProperty(option);

            if (opt.hasArg()) {
                if (opt.getValues() == null || opt.getValues().length == 0) {
                    opt.addValueForProcessing(value);
                }
            } else if (
                !("yes".equalsIgnoreCase(value)
                    || "true".equalsIgnoreCase(value)
                    || "1".equalsIgnoreCase(value))
            ) {
                // if the value is not yes, true or 1 then don't add the option to the CommandLine
                continue;
            }

            handleOption(opt);
            currentOption = null;
        }
    }
}
    
    '''
    
    Fix code 2:
    '''
    protected void processProperties(Properties properties) throws ParseException {
    if (properties == null) {
        return;
    }

    for (Enumeration<?> e = properties.propertyNames(); e.hasMoreElements();) {
        String option = e.nextElement().toString();

        Option opt = options.getOption(option);
        if (opt == null) {
            throw new UnrecognizedOptionException(
                "Default option wasn't defined", option
            );
        }

        // if the option is part of a group, check if another option of the group has been selected
        OptionGroup group = options.getOptionGroup(opt);
        boolean selected = (group != null && group.getSelected() != null);

        if (!cmd.hasOption(option) && !selected) {

            // get the value from the properties instance
            String value = properties.getProperty(option);

            if (opt.hasArg()) {
                if (opt.getValues() == null || opt.getValues().length == 0) {
                    try {
                        opt.addValueForProcessing(value);
                    } catch (RuntimeException exp) {
                        // if we cannot add the value don't worry about it
                    }
                }
            } else if (
                !("yes".equalsIgnoreCase(value)
                    || "true".equalsIgnoreCase(value)
                    || "1".equalsIgnoreCase(value))
            ) {
                // if the value is not yes, true or 1 then don't add the option to the CommandLine
                continue;
            }

            cmd.addOption(opt);
            updateRequiredOptions(opt);
        }
    }
}
    '''
    

    """

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_message}"),
            (
                "human",
                "Correct specification:\n{fix_speci}\n\n{buggy_code_text}",
            ),
        ]
    )

    
    messages = prompt_template.format_messages(
        system_message=system_message,
        fix_speci=fix_speci,
        buggy_code_text=buggy_code_text,
    )

    
    response = model.invoke(messages)
    
    
    usage = response.usage_metadata
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]

    
    cost = (
        input_tokens * PRICE_INPUT
        + output_tokens * PRICE_OUTPUT
    )

    
    return response.content, cost

def extract_fix_code_regex(text: str, index: int) -> str:
    pattern = rf"Fix code\s*{index}:\s*```[a-zA-Z]*\s*([\s\S]*?)```"
    match = re.search(pattern, text)

    if not match:
        raise ValueError(f"Could not find code block for Fix code {index}")

    code = match.group(1)

    
    code = code.strip()

    
    if not code.endswith("\n"):
        code += "\n"

    return code