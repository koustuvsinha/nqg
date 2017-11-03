--[[ Server deployment. Given an article, use each sentence to generate a question
    Runs a Waffle server https://github.com/benglard/waffle
--]]

require('onmt.init')
local app = require('waffle')

local function reportScore(name, scoreTotal, wordsTotal)
  print(string.format(name .. " AVG SCORE: %.4f, " .. name .. " PPL: %.4f",
                      scoreTotal / wordsTotal,
                      math.exp(-scoreTotal/wordsTotal)))
end

--[[ split input sentence into words
--]]
local function string2sent(sentence)
    local sent = {}
    for word in sentence:gmatch'([^%s]+)' do
        table.insert(sent, word)
    end
    return sent
end

-- options and configs
local opt = {}
opt['config'] = "config-trans"
opt['gpuid'] = 0
opt['fallback_to_cpu'] = False
opt['model'] = "model/840B.300d.600rnn_epoch15_25.06.t7"
opt['n_best'] = 1
opt['replace_unk'] = false
opt['time'] = false
opt['batch_size'] = 30
opt['beam_size'] = 5
opt['max_sent_length'] = 250
opt['src'] = ""
opt['phrase_table'] = ""


local requiredOptions = {
    "model",
    "src"
}

onmt.utils.Opt.init(opt, requiredOptions)
onmt.translate.Translator.init(opt)

-- predict question given a sentence
local function predict(srcTokens)

  local srcBatch = {}
  local srcWordsBatch = {}
  local srcFeaturesBatch = {}
  local sentId = 1
  local batchId = 1
  local prediction = {}

  local srcWords, srcFeats = onmt.utils.Features.extract(srcTokens)
  table.insert(srcBatch, srcTokens)
  table.insert(srcWordsBatch, srcWords)
  if #srcFeats > 0 then
      table.insert(srcFeaturesBatch, srcFeats)
  end

  local predBatch, info = onmt.translate.Translator.translate(srcWordsBatch, srcFeaturesBatch,
                                                                  tgtWordsBatch, tgtFeaturesBatch)
  local srcSent = table.concat(srcBatch[1], " ")
  local predSent = table.concat(predBatch[1], " ")
  prediction['pred'] = predSent
  prediction['score'] = info[1].score
  return prediction
end

-- server setup here.
app.post('/', function(req, res)
    local obj = req.body
    get sentences
    local preds = {}
    for i=1, #obj.sents do
        sent = obj.sents[i]
        words = string2sent(sent)
        question = predict(words)
        table.insert(preds, question)
    end
    -- send the sentences and scores back
    res.json(preds)
end)

app.listen()
