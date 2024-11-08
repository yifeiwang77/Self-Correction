import torch
import fastchat 

def load_conversation_template(template_name):
    conv_template = fastchat.model.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    
    return conv_template


class SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string, demo=None, sys=None):

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string
        self.demo = demo
        self.sys = sys
        self.update_sys()
        
    def update_sys(self):
        if self.sys is None:
            self.sys = ''
        if 'llama' in self.conv_template.name:
            self.conv_template.system = f"""<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.{self.sys}\n<</SYS>>\n\n"""
        elif 'falcon' in self.conv_template.name:
            self.conv_template.system = f"""{self.sys}"""
        else:    
            self.conv_template.system = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {self.sys}"""

    def update_demo(self):
        if self.demo is None:
            return
        for q, a in self.demo:
            self.conv_template.append_message(self.conv_template.roles[0], q)
            self.conv_template.append_message(self.conv_template.roles[1], a)
    
    def get_prompt(self, adv_string=None):
        self.conv_template.messages = []
        if adv_string is not None:
            self.adv_string = adv_string

        self.update_demo()
        if self.adv_string is None:
            self.conv_template.append_message(self.conv_template.roles[0], f"{self.instruction}")
        else:
            self.conv_template.append_message(self.conv_template.roles[0], f"{self.instruction} {self.adv_string}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()
        # print(prompt)
        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids

        if self.conv_template.name == 'llama-2' or self.conv_template.name == 'falcon':
            self.conv_template.messages = []
            self.update_sys()
            self.update_demo()
            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            if self.demo is None:
                self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))
            else:
                self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)-1))
                
            separator = ' ' if self.instruction else ''
            if self.adv_string is None:
                self.conv_template.update_last_message(f"{self.instruction}")
            else:
                self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            if self.demo is None:
                self._control_slice = slice(self._goal_slice.stop, len(toks))
            else:
                self._control_slice = slice(self._goal_slice.stop, len(toks)-1)

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            # print(self.tokenizer.decode(toks[self._control_slice]))
            # assert False
            
            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-2)
            self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-3)



        else:
            python_tokenizer = False or self.conv_template.name == 'oasst_pythia'
            try:
                encoding.char_to_token(len(prompt)-1)
            except:
                python_tokenizer = True

            if python_tokenizer:
                # This is specific to the vicuna and pythia tokenizer and conversation prompt.
                # It will not work with other tokenizers or prompts.
                self.conv_template.messages = []
                self.update_sys()
                self.update_demo()
                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                self.conv_template.update_last_message(f"{self.instruction}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)-1))

                separator = ' ' if self.instruction else ''
                if self.adv_string is None:
                    self.conv_template.update_last_message(f"{self.instruction}")
                else:
                    self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._control_slice = slice(self._goal_slice.stop, len(toks)-1)

                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-1)
                self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-2)
            else:
                raise NotImplementedError
                self._system_slice = slice(
                    None, 
                    encoding.char_to_token(len(self.conv_template.system))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)
                )
                self._goal_slice = slice(
                    encoding.char_to_token(prompt.find(self.instruction)),
                    encoding.char_to_token(prompt.find(self.instruction) + len(self.instruction))
                )
                self._control_slice = slice(
                    encoding.char_to_token(prompt.find(self.adv_string)),
                    encoding.char_to_token(prompt.find(self.adv_string) + len(self.adv_string))
                )
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)),
                    encoding.char_to_token(prompt.find(self.target) + len(self.target))
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)) - 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1
                )

        self.conv_template.messages = []

        return prompt
    
    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids

