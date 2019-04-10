class SquadExample(object):

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):

        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        # Note: The start and end positions stores
        # word based indexing positions
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):

        return self.__repr__()

    def __repr__(self):
        s = "*************** Sample **************************"
        s += "\n  - qas_id: {}".format(self.qas_id)
        s += "\n  - question_text: {}".format(self.question_text)
        s += "\n  - answer_text  : {}".format(self.orig_answer_text)
        if self.start_position:
            s += "\n  - start_position: {}".format(self.start_position)
        if self.start_position:
            s += "\n  - end_position: {}".format(self.end_position)
            s += "\n  - doc_tokens: \n\n     {}" \
                 "\n\n************************************************\n\n".format(" ".join(self.doc_tokens))
        return s


class InputFeatures(object):
    """
    A single set of rfeatures of data
    """

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = "\n"
        s += "unique_id: {} \n".format(unique_id)
        s += "example_index: {} \n".format(example_index)
        s += "doc_span_index: {} \n".format(doc_span_index)
        s += "tokens: {} \n".format(" ".join(tokens))
        s += "tokens_to_origin_map: {} \n".format(" ".join([
            "{}:{}".format(x, y) for (x, y) in token_to_orig_map.items()]))
        s += "token_is_max_content: {} \n".format(" ".join([
            "{}:{}".format(x, y) for (x, y) in token_is_max_context.items()]))
        s += "input_ids: {} \n".format(" ".join([str(x) for x in input_ids]))
        s += "input_mask: {}\n".format(" ".join([str(x) for x in input_mask]))
        s += "segment_ids: {}\n".format(" ".join([str(x) for x in segment_ids]))
        # only when training
        s += "answer_text : {}\n".format(" ".join(tokens[start_position:(end_position + 1)]))
        s += "start_position: {}\n".format(start_position)
        s += "end_position : {}\n ".format(end_position)
        return s