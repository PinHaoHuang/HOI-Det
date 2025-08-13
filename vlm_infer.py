from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch



def ask_vlm(
    image_path: str,
    question: str,
    model,
    tokenizer,
    image_processor,
    device,
    conv_template: str = "llava_llama_3",
    max_new_tokens: int = 256
):
    """
    Run a vision-language model (e.g., LLaVA) on a given image and question,
    returning the model's text output.

    Args:
        image_path (str): Path to the input image file.
        question (str): The textual prompt/question to ask, typically
                        including the `DEFAULT_IMAGE_TOKEN` as needed.
        model (torch.nn.Module): The loaded VLM (LLaVA-style) model.
        tokenizer: The tokenizer corresponding to your LLaVA model.
        image_processor: A function/class that preprocesses raw images to
                         model-ready tensors.
        conv_templates (dict): Dictionary of conversation templates.
        device: The device (CPU/GPU) to run inference on.
        conv_template (str): Which conversation template key to use from conv_templates.
        DEFAULT_IMAGE_TOKEN (str): The special token for "image", e.g. "<image>".
        IMAGE_TOKEN_INDEX (int): The vocab index corresponding to <image>.
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: The model's text response.

    Example usage:
        response_text = ask_vlm(
            "/path/to/image.jpg",
            "The left hand is holding: A. Mug, B. Phone, C. Coffee Pot, D. None?\nAnswer with format LETTER_OF_CHOICE-CONFIDENCE_SCORE",
            model, tokenizer, image_processor, conv_templates, device
        )
        print("Model says:", response_text)
    """

    # 1) Load image
    image = Image.open(image_path).convert("RGB")

    # 2) Process image using your custom function
    #    Typically returns a list of torch.Tensors or a single Tensor
    image_tensors = process_images([image], image_processor, model.config)
    #   Convert to half-precision & move to device
    image_tensors = [
        img_tensor.to(dtype=torch.float16, device=device)
        for img_tensor in image_tensors
    ]

    # 3) Prepare conversation template

    print(question)
    question = DEFAULT_IMAGE_TOKEN + question
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.tokenizer = tokenizer

    # Append user question (the conversation frameworks typically have roles = ["user", "assistant"])
    conv.append_message(conv.roles[0], question)
    # The assistant's reply is None for now
    conv.append_message(conv.roles[1], None)

    # 4) Build the full text prompt from the conversation template
    prompt_question = conv.get_prompt()

    # 5) Tokenize text prompt, inserting the image token properly
    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    # 6) Prepare image sizes if the model needs them
    image_size = image.size  # (width, height)
    image_sizes = [image_size]  # batch of size 1

    # 7) Generate model's response
    generation_output = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=max_new_tokens,
    )

    # 8) Decode the response
    text_output = tokenizer.batch_decode(
        generation_output, skip_special_tokens=True
    )[0]

    
    print(text_output)
    return text_output