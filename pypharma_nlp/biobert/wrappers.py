import numpy as np
import os
import warnings
from nltk.tokenize.treebank import TreebankWordDetokenizer
with warnings.catch_warnings():
    warnings.simplefilter(action="ignore")
    from pypharma_nlp.biobert import modeling
    from pypharma_nlp.biobert import tokenization
    from pypharma_nlp.biobert import classification
    from pypharma_nlp.biobert import ner
    from pypharma_nlp.biobert import ner_wrappers
    from pypharma_nlp.biobert import relation_extraction
    from pypharma_nlp.biobert import question_answering
    tf = classification.tf

#def _ner_input_fn_builder(features, seq_length, is_training, drop_remainder):
#    all_input_ids = []
#    all_input_mask = []
#    all_segment_ids = []
#    all_label_ids = []
#
#    for feature in features:
#        all_input_ids.append(feature.input_ids)
#        all_input_mask.append(feature.input_mask)
#        all_segment_ids.append(feature.segment_ids)
#        all_label_ids.append(feature.label_id)
#
#    def input_fn(params):
#        """The actual input function."""
#        batch_size = params["batch_size"]
#
#        num_examples = len(features)
#
#        # This is for demo purposes and does NOT scale to large data sets. We do
#        # not use Dataset.from_generator() because that uses tf.py_func which is
#        # not TPU compatible. The right way to load data is with
#        # TFRecordReader.
#        d = tf.data.Dataset.from_tensor_slices({
#                "input_ids":
#                        tf.constant(
#                                all_input_ids, shape=[
#                                    num_examples, seq_length],
#                                dtype=tf.int32),
#                "input_mask":
#                        tf.constant(
#                                all_input_mask,
#                                shape=[num_examples, seq_length],
#                                dtype=tf.int32),
#                "segment_ids":
#                        tf.constant(
#                                all_segment_ids,
#                                shape=[num_examples, seq_length],
#                                dtype=tf.int32),
#                "label_ids":
#                        tf.constant(all_label_ids, shape=[num_examples], 
#                            dtype=tf.int32),
#        })
#
#        if is_training:
#            d = d.repeat()
#            d = d.shuffle(buffer_size=100)
#
#        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
#        return d
#
#    return input_fn

def _question_answering_input_fn_builder(features, seq_length, is_training, 
    drop_remainder):
    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with
        # TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
                "input_ids":
                        tf.constant(
                                all_input_ids, shape=[
                                    num_examples, seq_length],
                                dtype=tf.int32),
                "input_mask":
                        tf.constant(
                                all_input_mask,
                                shape=[num_examples, seq_length],
                                dtype=tf.int32),
                "segment_ids":
                        tf.constant(
                                all_segment_ids,
                                shape=[num_examples, seq_length],
                                dtype=tf.int32),
                "unique_ids":
                        tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


CLASSIFICATION_PROCESSORS = {
    "ade" : classification.AdeProcessor, 
    "hoc" : classification.HocProcessor, 
}


RELATION_EXTRACTION_PROCESSORS = {
    "gad": relation_extraction.BioBERTProcessor,
    "polysearch": relation_extraction.BioBERTProcessor,
    "mirnadisease": relation_extraction.BioBERTProcessor,
    "euadr": relation_extraction.BioBERTProcessor,
    "chemprot": relation_extraction.BioBERTChemprotProcessor,
    "ddi13": relation_extraction.BioBERTDDIProcessor,
}


class BioBertWrapper(object):

    """A wrapper object to encapsulate BioBert's functionality."""

    def __init__(self):
        self._estimator = None
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
        self._do_lower_case = True
        self._max_seq_length = 128
        self._do_lower_case = True
        self._train_batch_size = 32
        self._eval_batch_size = 8
        self._predict_batch_size = 8
        self._learning_rate = 5e-5
        self._num_train_epochs = 3.0
        self._warmup_proportion = 0.1
        self._save_checkpoints_steps = 1000
        self._iterations_per_loop = 1000
        self._use_tpu = False
        self._tpu_name = None
        self._tpu_zone = None
        self._gcp_project = None
        self._master = None
        self._num_tpu_cores = 8
        self._tokenizer = None
        self._task_type = None
        self._estimator = None
        self._task_name = None
        self._labels = None
        self._doc_stride = None
        self._max_query_length = None
        self._n_best_size = None
        self._max_answer_length = None
        self._version_2_with_negative = None
        self._null_score_diff_threshold = None
        self._model_fn = None

    def _build_classification(self, bert_config, init_checkpoint):
        self._do_lower_case = True
        self._max_seq_length = 128
        self._train_batch_size = 32
        self._eval_batch_size = 8
        self._predict_batch_size = 8
        self._learning_rate = 5e-5
        self._num_train_epochs = 3.0
        self._warmup_proportion = 0.1
        self._save_checkpoints_steps = 1000
        self._iterations_per_loop = 1000
        self._use_tpu = False
        self._num_tpu_cores = 8
        processor = CLASSIFICATION_PROCESSORS[self._task_name]()
        self._labels = processor.get_labels()

        self._model_fn = classification.model_fn_builder(
            bert_config=bert_config,
            num_labels=len(self._labels),
            init_checkpoint=init_checkpoint,
            learning_rate=self._learning_rate,
            num_train_steps=0,
            num_warmup_steps=0,
            use_tpu=self._use_tpu,
            use_one_hot_embeddings=self._use_tpu)

    def _build_question_answering(self, bert_config, init_checkpoint):
        self._do_lower_case = True
        self._max_seq_length = 384
        self._train_batch_size = 32
        self._eval_batch_size = 8
        self._predict_batch_size = 8
        self._learning_rate = 5e-5
        self._num_train_epochs = 3.0
        self._warmup_proportion = 0.1
        self._save_checkpoints_steps = 1000
        self._iterations_per_loop = 1000
        self._use_tpu = False
        self._num_tpu_cores = 8
        self._doc_stride = 128
        self._max_query_length = 64
        self._n_best_size = 20
        self._max_answer_length = 30

        self._model_fn = question_answering.model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=init_checkpoint,
            learning_rate=self._learning_rate,
            num_train_steps=0,
            num_warmup_steps=0,
            use_tpu=self._use_tpu,
            use_one_hot_embeddings=self._use_tpu)

    def _build_relation_extraction(self, bert_config, init_checkpoint):
        self._do_lower_case = True
        self._max_seq_length = 128
        self._train_batch_size = 32
        self._eval_batch_size = 8
        self._predict_batch_size = 8
        self._learning_rate = 5e-5
        self._num_train_epochs = 3.0
        self._warmup_proportion = 0.1
        self._save_checkpoints_steps = 1000
        self._iterations_per_loop = 1000
        self._use_tpu = False
        self._num_tpu_cores = 8
        processor = RELATION_EXTRACTION_PROCESSORS[self._task_name]()
        self._labels = processor.get_labels()
        
        self._model_fn = relation_extraction.model_fn_builder(
            bert_config=bert_config,
            num_labels=len(self._labels),
            init_checkpoint=init_checkpoint,
            learning_rate=self._learning_rate,
            num_train_steps=0,
            num_warmup_steps=0,
            use_tpu=self._use_tpu,
            use_one_hot_embeddings=self._use_tpu)

    def _build_ner(self, bert_config, init_checkpoint):
        self._do_lower_case = True
        self._max_seq_length = 128
        self._train_batch_size = 32
        self._eval_batch_size = 8
        self._predict_batch_size = 8
        self._learning_rate = 5e-5
        self._num_train_epochs = 3.0
        self._warmup_proportion = 0.1
        self._save_checkpoints_steps = 1000
        self._iterations_per_loop = 1000
        self._use_tpu = False
        self._num_tpu_cores = 8
        processor = ner.NerProcessor()
        self._labels = processor.get_labels()

        self._model_fn = ner_wrappers.model_fn_builder(
            bert_config=bert_config,
            num_labels=len(self._labels) + 1,
            init_checkpoint=init_checkpoint,
            learning_rate=self._learning_rate,
            num_train_steps=0,
            num_warmup_steps=0,
            use_tpu=self._use_tpu,
            use_one_hot_embeddings=self._use_tpu, 
            max_seq_length=self._max_seq_length)
        
    def build(self, task_type, task_name, model_directory, init_checkpoint):
        """Setup the wrapper model based on a checkpoint and a task"""

        self._task_type = task_type
        vocab_file = os.path.join(model_directory, "vocab.txt")
        bert_config_file = os.path.join(
          model_directory, "bert_config.json")

        tokenization.validate_case_matches_checkpoint(self._do_lower_case,
            init_checkpoint)

        bert_config = modeling.BertConfig.from_json_file(
          bert_config_file)

        if self._max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (self._max_seq_length, bert_config.max_position_embeddings))

        self._task_name = task_name.lower()

        self._tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=self._do_lower_case)

        tpu_cluster_resolver = None
        if self._use_tpu and self._tpu_name:
            tpu_cluster_resolver = \
                tf.contrib.cluster_resolver.TPUClusterResolver(
                self._tpu_name, zone=self._tpu_zone, project=self._gcp_project)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=self._master,
            model_dir=None, 
            save_checkpoints_steps=self._save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self._iterations_per_loop,
                num_shards=self._num_tpu_cores,
                per_host_input_for_training=is_per_host))

        train_examples = None

        if task_type == "classification":
            self._build_classification(
              bert_config, init_checkpoint)
        elif task_type == "relation_extraction":
            self._build_relation_extraction(bert_config, init_checkpoint)
        elif task_type == "ner":
            self._build_ner(bert_config, init_checkpoint)
        elif task_type == "question_answering":
            self._build_question_answering(bert_config, init_checkpoint)
        else:
            raise ValueError("Unknown task type: '%s'." % task_type)

        # If TPU is not available, this will fall back to normal Estimator on
        # CPU or GPU.
        self._estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=self._use_tpu,
            model_fn=self._model_fn,
            config=run_config,
            train_batch_size=self._train_batch_size,
            eval_batch_size=self._eval_batch_size,
            predict_batch_size=self._predict_batch_size)

    def classify(self, sentences):
        examples = [classification.InputExample(guid=str(i + 1),
            text_a=sentences[i], text_b=None, label=self._labels[0]) for i in
            range(len(sentences))]
        num_actual_examples = len(examples)
        if self._use_tpu:
            # TPU requires a fixed batch size for all batches, therefore
            # the number of examples must be a multiple of the batch size,
            # or else examples will get dropped. So we pad with fake
            # examples which are ignored later on.
            while len(
              examples) % self._predict_batch_size != 0:
                examples.append(PaddingInputExample())

        features = classification.convert_examples_to_features(
            examples, self._labels, self._max_seq_length,
            self._tokenizer)

        predict_drop_remainder = True if self._use_tpu else False
        predict_input_fn = classification.input_fn_builder(features,
            self._max_seq_length, False, predict_drop_remainder)

        result = self._estimator.predict(input_fn=predict_input_fn)
        predictions = np.array([r["probabilities"] for r in result])
        return predictions
        
    def extract_relations(self, sentences):
        examples = [relation_extraction.InputExample(guid=str(i),
            text_a=sentences[i], text_b=None, label=self._labels[0]) for i in 
            range(len(sentences))]
        num_actual_examples = len(examples)
        if self._use_tpu:
            # TPU requires a fixed batch size for all batches, therefore
            # the number of examples must be a multiple of the batch size,
            # or else examples will get dropped. So we pad with fake
            # examples which are ignored later on.
            while len(
              examples) % self._predict_batch_size != 0:
                examples.append(PaddingInputExample())

        features = relation_extraction.convert_examples_to_features(
            examples, self._labels, self._max_seq_length,
            self._tokenizer)

        predict_drop_remainder = True if self._use_tpu else False
        predict_input_fn = relation_extraction.input_fn_builder(features,
            self._max_seq_length, False, predict_drop_remainder)

        result = self._estimator.predict(input_fn=predict_input_fn)
        predictions = np.array([r["probabilities"] for r in result])
        return predictions

    def answer(self, questions, contexts):
        examples = [
            question_answering.SquadExample(
                qas_id=str(i),
                question_text=questions[i],
                doc_tokens=contexts[i].split(),
                orig_answer_text="",
                start_position=0,
                end_position=0,
                is_impossible=False
            ) for i in range(len(questions))]
        features = []

        def add_feature(feature):
            features.append(feature)

        question_answering.convert_examples_to_features(
            examples=examples,
            tokenizer=self._tokenizer,
            max_seq_length=self._max_seq_length,
            doc_stride=self._doc_stride,
            max_query_length=self._max_query_length,
            is_training=False,
            output_fn=add_feature)

        all_results = []

        predict_drop_remainder = True if self._use_tpu else False
        predict_input_fn = _question_answering_input_fn_builder(features,
            self._max_seq_length, False, predict_drop_remainder)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        all_results = []
        for result in self._estimator.predict(
            predict_input_fn, yield_single_examples=True):
            unique_id = int(result["unique_ids"])
            start_logits = [float(x) for x in result["start_logits"].flat]
            end_logits = [float(x)
                      for x in result["end_logits"].flat]
            all_results.append(
                question_answering.RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))

        predictions, n_best, scores_diff = self._decode_predictions(examples, 
            features, all_results)
        return predictions["0"]
        
    def _decode_predictions(self, all_examples, all_features, all_results):
        
        example_index_to_features = question_answering.collections.defaultdict(
            list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)
    
        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result
    
        _PrelimPrediction = question_answering.collections.namedtuple(
            "PrelimPrediction", ["feature_index", "start_index", "end_index", 
            "start_logit", "end_logit"])
    
        all_predictions = question_answering.collections.OrderedDict()
        all_nbest_json = question_answering.collections.OrderedDict()
        scores_diff_json = question_answering.collections.OrderedDict()
    
        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]
    
            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000    # large and positive
            min_null_feature_index = 0 # the paragraph slice with min mull score
            null_start_logit = 0 # the start logit at the slice with min null 
                                 # score
            null_end_logit = 0 # the end logit at the slice with min null score
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                start_indexes = question_answering._get_best_indexes(
                    result.start_logits, self._n_best_size)
                end_indexes = question_answering._get_best_indexes(
                    result.end_logits, self._n_best_size)
                # if we could have irrelevant answers, get the min score of 
                # irrelevant
                if self._version_2_with_negative:
                    feature_null_score = result.start_logits[0] + \
                        result.end_logits[0]
                    if feature_null_score < score_null:
                        score_null = feature_null_score
                        min_null_feature_index = feature_index
                        null_start_logit = result.start_logits[0]
                        null_end_logit = result.end_logits[0]
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, 
                        # e.g., predict
                        # that the start of the span is in the question. We 
                        # throw out all
                        # invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, 
                            False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > self._max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[
                                    start_index],
                                end_logit=result.end_logits[end_index]))
    
            if self._version_2_with_negative:
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=min_null_feature_index,
                        start_index=0,
                        end_index=0,
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True)
    
            _NbestPrediction = question_answering.collections.namedtuple(
                    "NbestPrediction", ["text", "start_logit", "end_logit"])
    
            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= self._n_best_size:
                    break
                feature = features[pred.feature_index]
                if pred.start_index > 0:    # this is a non-null prediction
                    tok_tokens = feature.tokens[pred.start_index:(
                        pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[orig_doc_start:(
                        orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)
    
                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")
    
                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)
    
                    final_text = question_answering.get_final_text(tok_text, 
                        orig_text, self._do_lower_case)
                    if final_text in seen_predictions:
                        continue
    
                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True
    
                nbest.append(
                        _NbestPrediction(
                                text=final_text,
                                start_logit=pred.start_logit,
                                end_logit=pred.end_logit))
    
            # if we didn't inlude the empty option in the n-best, inlcude it
            if self._version_2_with_negative:
                if "" not in seen_predictions:
                    nbest.append(
                            _NbestPrediction(
                                    text="", start_logit=null_start_logit,
                                    end_logit=null_end_logit))
            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(text="empty", start_logit=0.0, 
                    end_logit=0.0))
    
            assert len(nbest) >= 1
    
            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry
    
            probs = question_answering._compute_softmax(total_scores)
    
            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = question_answering.collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)
    
            assert len(nbest_json) >= 1
    
            if not self._version_2_with_negative:
                all_predictions[example.qas_id] = nbest_json[0]["text"]
            else:
                # predict "" iff the null score - the score of best non-null > 
                # threshold
                score_diff = score_null - best_non_null_entry.start_logit - (
                        best_non_null_entry.end_logit)
                scores_diff_json[example.qas_id] = score_diff
                if score_diff > self._null_score_diff_threshold:
                    all_predictions[example.qas_id] = ""
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text
    
            all_nbest_json[example.qas_id] = nbest_json
    
        return all_predictions, all_nbest_json, scores_diff_json

    def extract_entities(self, sentences):
        
        # Detokenize token lists if needed
        if type(sentences) == list:
            detokenizer = TreebankWordDetokenizer()
            sentences = [detokenizer.detokenize(s) for s in sentences]
            
        examples = [
            ner.InputExample(
                i, 
                sentences[i], 
                " ".join(["O"] * len(sentences[i].split(" "))), 
            ) for i in range(len(sentences))]
        features, tokens = ner_wrappers.convert_examples_to_features(
            examples, self._labels, self._max_seq_length, self._tokenizer, 
            mode="test")
                            
        if self._use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if self._use_tpu else False
        predict_input_fn = ner_wrappers.input_fn_builder(
            features=features,
            seq_length=self._max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        prf = self._estimator.evaluate(input_fn=predict_input_fn, steps=None)

        result = self._estimator.predict(input_fn=predict_input_fn)
        predictions = []
        for i, prediction in enumerate(result):
            labels = []
            length = len(tokens[i])
            print(prediction["prediction"], tokens[i], length)
            for id in prediction["prediction"][:length]:
                if id != 0:
                    label = self._labels[id - 1]
                else:
                    label = self._labels[2]
                labels.append(label)
            prediction["prediction"] = labels
            predictions.append(prediction)
        return predictions

        
if __name__ == "__main__":
    
    wrapper = BioBertWrapper()
    wrapper.build("ner", "ner", "models/biobert_v1.0_pubmed_pmc/", 
        "checkpoints/biobert_bioasq/model.ckpt-1988")
    print(wrapper.extract_entities(["another example"]))
    
    wrapper = BioBertWrapper()
    wrapper.build("relation_extraction", "chemprot", 
        "models/biobert_v1.0_pubmed_pmc/", 
        "checkpoints/biobert_bioasq/model.ckpt-1988")
    print(wrapper.extract_relations(["another example"]))

    wrapper = BioBertWrapper()
    wrapper.build("classification", "ade", "models/biobert_v1.0_pubmed_pmc/", 
        "checkpoints/biobert_bioasq/model.ckpt-1988")
    print(wrapper.classify(["another example"]))
    
    wrapper = BioBertWrapper()
    wrapper.build("question_answering", "ade", 
        "models/biobert_v1.0_pubmed_pmc/", 
        "checkpoints/biobert_bioasq/model.ckpt-1988")
    
    context = "Border Collies require considerably more daily physical exercise and mental stimulation than many other breeds. The Border Collie is widely considered to be the most intelligent dog breed. The Border Collie ranks 1st in Stanley Coren's The Intelligence of Dogs, being part of the top 10 brightest dogs. Although the primary role of the Border Collie is to herd livestock, the breed is becoming increasingly popular as a companion animal."
    print("CONTEXT:\n\n%s\n" % context)
    print("Ask me a question abou this context:\nQUESTION:")
    while True:
        question = input()
        print("ANSWER: %s\n" % wrapper.answer([question], [context]))
