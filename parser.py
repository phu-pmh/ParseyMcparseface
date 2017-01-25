from collections import OrderedDict
import subprocess
from subprocess import PIPE
ROOT_DIR = "models/syntaxnet"
PARSER_EVAL = "bazel-bin/syntaxnet/parser_eval"
MODEL_DIR = "syntaxnet/models/parsey_mcparseface"

def open_parser_eval(args):
  return subprocess.Popen(
    [PARSER_EVAL] + args,
    cwd=ROOT_DIR,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
  )

def send_input(process, sent):
  
  response = b""
  response = process.communicate(input = sent.encode("utf8"))[0] 
  
  return response.decode("utf8")

# Open the part-of-speech tagger.
pos_tagger = open_parser_eval([
    "--input=stdin",
    "--output=stdout-conll",
    "--hidden_layer_sizes=64",
    "--arg_prefix=brain_tagger",
    "--graph_builder=structured",
    "--task_context=" + MODEL_DIR + "/context.pbtxt",
    "--model_path=" + MODEL_DIR + "/tagger-params",
    "--slim_model",
    "--batch_size=1024",
    "--alsologtostderr",
  ])

# Open the syntactic dependency parser.
dependency_parser = open_parser_eval([
    "--input=stdin-conll",
    "--output=stdout-conll",
    "--hidden_layer_sizes=512,512",
    "--arg_prefix=brain_parser",
    "--graph_builder=structured",
    "--task_context=" + MODEL_DIR + "/context.pbtxt",
    "--model_path=" + MODEL_DIR + "/parser-params",
    "--slim_model",
    "--batch_size=1024",
    "--alsologtostderr",
  ])

def split_tokens(parse):
  # Format the result.
  def format_token(line):
    x = OrderedDict(zip(
     ["index", "token", "unknown1", "label", "pos", "unknown2", "parent", "relation", "unknown3", "unknown4"],
     line.split("\t")
    ))
    x["index"] = int(x["index"])
    x["parent"] = int(x["parent"])
    del x["unknown1"]
    del x["unknown2"]
    del x["unknown3"]
    del x["unknown4"]
    return x
                                   
  return [
    format_token(line)
    for line in parse.strip().split("\n")
  ]

def parse_sentence(sentence):
  if "\n" in sentence or "\r" in sentence:
    raise ValueError()

  # Do POS tagging.
  pos_tags = send_input(pos_tagger, sentence + "\n")

  # Do dependency parsing.
  dependency_parse = send_input(dependency_parser, pos_tags)
  # Make a tree.
  dependency_parse = split_tokens(dependency_parse)
   
  indexed_words = [ tok["token"]+'-'+ str(tok["index"]) for tok in dependency_parse]
  print('indexed words : ' , indexed_words) 
  
  dep_list = []
  pos_list = []
  token_list = []
  for i in range(len(dependency_parse)):
     dep_list.append((dependency_parse[i]["relation"], indexed_words[dependency_parse[i]["parent"]-1], indexed_words[i]))
     pos_list.append(dependency_parse[i]['label'])
     token_list.append(dependency_parse[i]['token'])

  #to return as a dictionary   
  d = {}
  d['pos'] = pos_list
  d['dependency'] = dep_list
  d['tokens'] = token_list
  return d


if __name__ == "__main__":
  import sys, pprint
  sentence = "I want to eat sushi."	
  print("Printing sent: " , sentence)
  parse_dict = parse_sentence(sentence.strip())
  print("getting dep : " , parse_dict)