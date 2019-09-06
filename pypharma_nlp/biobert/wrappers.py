import numpy as np
import os
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action="ignore")
    from pypharma_nlp.biobert import modeling
    from pypharma_nlp.biobert import tokenization
    from pypharma_nlp.biobert.classification import *
    #from pypharma_nlp.biobert import classification


class AdeProcessor(classification.DataProcessor):

    """Processor for the ADE Corpus V2."""
    
    def __init__(self, data_directory):
        self._data_directory = data_directory
        self._examples = {
            "train" : [], 
            "dev" : [], 
            "test" : [], 
        }
        self._build_examples()
    
    def _get_random_subset(self):
        import random
        random_number = random.uniform(0, 1)
        if random_number <= 0.7:
            subset = "train"
        elif random_number <= 0.85:
            subset = "dev"
        else:
            subset = "test"
        return subset
    
    def _build_examples(self):
        ade_corpus.download_source_data(self._data_directory)
        import random
        random.seed(9999)
        for pmid, sentences, labels in ade_corpus.get_classification_examples(
            self._data_directory):
            subset = self._get_random_subset()
            count = 1
            for i in range(len(sentences)):
                guid = "%s-%d" % (pmid, count)
                text_a = tokenization.convert_to_unicode(sentences[i])
                label = tokenization.convert_to_unicode(labels[i])
                example = InputExample(guid="%s_%d" % (pmid, count), 
                    text_a=text_a, text_b=None, label=label)
                self._examples[subset].append(example)
                count += 1
                        
    def get_train_examples(self):
        """See base class."""
        return self._examples["train"]

    def get_dev_examples(self):
        """See base class."""
        return self._examples["dev"]

    def get_test_examples(self):
        """See base class."""
        return self._examples["test"]
    
    def get_labels(self):
        """See base class."""
        labels = [
            "Neg", 
            "AE"
        ]
        return labels


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
        self._processor = None
        self._label_list = None
        self._tokenizer = None
        self._estimator = None


    def build(self, task_name, model_directory, data_dir, output_dir):
        
        """Setup the wrapper model based on a checkpoint and a task"""
                
        processors = {
            "cola": ColaProcessor,
            "mnli": MnliProcessor,
            "mrpc": MrpcProcessor,
            "xnli": XnliProcessor,
            "ade" : AdeProcessor, 
            "hoc" : HocProcessor, 
        }

        vocab_file = os.path.join(model_directory, "vocab.txt")
        bert_config_file = os.path.join(model_directory, "bert_config.json")
        init_checkpoint = os.path.join(model_directory, "biobert_model.ckpt")
        
        tokenization.validate_case_matches_checkpoint(self._do_lower_case, 
            init_checkpoint)

        bert_config = modeling.BertConfig.from_json_file(bert_config_file)

        if self._max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (self._max_seq_length, bert_config.max_position_embeddings))

        tf.gfile.MakeDirs(output_dir)

        task_name = task_name.lower()

        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))

        self._processor = processors[task_name](data_dir)

        self._label_list = self._processor.get_labels()

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
            model_dir=output_dir,
            save_checkpoints_steps=self._save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self._iterations_per_loop,
                num_shards=self._num_tpu_cores,
                per_host_input_for_training=is_per_host))

        train_examples = None
        num_train_steps = None
        num_warmup_steps = None
        
        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(self._label_list),
            init_checkpoint=init_checkpoint,
            learning_rate=self._learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=self._use_tpu,
            use_one_hot_embeddings=self._use_tpu)

        # If TPU is not available, this will fall back to normal Estimator on 
        # CPU or GPU.
        self._estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=self._use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=self._train_batch_size,
            eval_batch_size=self._eval_batch_size,
            predict_batch_size=self._predict_batch_size)

    
    def predict(self, sentences):
        predict_examples = [InputExample(guid=str(i + 1), text_a=sentences[i], 
            text_b=None, label="AE") for i in range(len(sentences))]
        num_actual_predict_examples = len(predict_examples)
        if self._use_tpu:
            # TPU requires a fixed batch size for all batches, therefore 
            # the number of examples must be a multiple of the batch size, 
            # or else examples will get dropped. So we pad with fake 
            # examples which are ignored later on.
            while len(predict_examples) % self._predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())
        
        #predict_file = os.path.join(output_dir, "predict.tf_record")
        features = convert_examples_to_features(predict_examples, 
            self._label_list, self._max_seq_length, self._tokenizer)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("    Num examples = %d (%d actual, %d padding)", 
            len(predict_examples), num_actual_predict_examples, 
            len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("    Batch size = %d", self._predict_batch_size)
        
        predict_drop_remainder = True if self._use_tpu else False
        predict_input_fn = input_fn_builder(features, self._max_seq_length,
            False, predict_drop_remainder)

        result = self._estimator.predict(input_fn=predict_input_fn)
        predictions = np.array([r["probabilities"] for r in result])
        return predictions
        

if __name__ == "__main__":
    wrapper = BioBertWrapper()
    wrapper.build("ade", "models/biobert_v1.0_pubmed/", "data/ade_corpus", 
        "output")
    print(wrapper.predict(["whatever fuck"]))
