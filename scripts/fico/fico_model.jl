module FicoClassifier
export predict, FICO_CLASSIFIER
import MLJModelInterface
const MMI = MLJModelInterface

#tree_model = @load FICO_CLASSIFIER pkg=GeneticCounterfactual
mutable struct FICO_CLASSIFIER <: MMI.Deterministic
    lambda::Float64
end


function MMI.clean!(model::FICO_CLASSIFIER)
    warning = ""
    if model.lambda < 0
        warning *= "Need lambda ≥ 0. Resetting lambda=0. "
        model.lambda = 0
    end
    return warning
end

# keyword constructor
function FICO_CLASSIFIER(; lambda=0.0)
    model = FICO_CLASSIFIER(lambda)
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end


function MMI.fit(model::FICO_CLASSIFIER, verbosity, X, y)
    
    fitresult = risk_model
    cache = nothing
    report = nothing
    return fitresult, cache, report
end


function predict(model::FICO_CLASSIFIER, Xnew)

    outcomes = Array{Float32}(undef, size(Xnew)[1])
    for i in 1:size(Xnew)[1]
        outcomes[i] = 1 - risk_model(Xnew[i, :])
    end
    return outcomes
end


## transform the entity to vector (bucketize)
# convert an entity to vector of sublayers
function entity_to_vectors(entity)
    vect = Array{Int64,1}(undef, 23)
    vect[1] =  feature_to_vector(entity, "ExternalRiskEstimate")
    vect[2] =  feature_to_vector(entity, "MSinceOldestTradeOpen")
    vect[3] =  feature_to_vector(entity, "MSinceMostRecentTradeOpen")
    vect[4] =  feature_to_vector(entity, "AverageMInFile")
    vect[5] =  feature_to_vector(entity, "NumSatisfactoryTrades")
    vect[6] =  feature_to_vector(entity, "NumTrades60Ever2DerogPubRec")
    vect[7] =  feature_to_vector(entity, "NumTrades90Ever2DerogPubRec")
    vect[8] =  feature_to_vector(entity, "NumTotalTrades")
    vect[9] =  feature_to_vector(entity, "NumTradesOpeninLast12M")
    vect[10] =  feature_to_vector(entity, "PercentTradesNeverDelq")
    vect[11] =  feature_to_vector(entity, "MSinceMostRecentDelq")
    vect[12] =  feature_to_vector(entity, "MaxDelq2PublicRecLast12M")
    vect[13] =  feature_to_vector(entity, "MaxDelqEver")
    vect[14] =  feature_to_vector(entity, "PercentInstallTrades")
    vect[15] =  feature_to_vector(entity, "NetFractionInstallBurden")
    vect[16] =  feature_to_vector(entity, "NumInstallTradesWBalance")
    vect[17] =  feature_to_vector(entity, "MSinceMostRecentInqexcl7days")
    vect[18] =  feature_to_vector(entity, "NumInqLast6M")
    vect[19] =  feature_to_vector(entity, "NumInqLast6Mexcl7days")
    vect[20] =  feature_to_vector(entity, "NetFractionRevolvingBurden")
    vect[21] =  feature_to_vector(entity, "NumRevolvingTradesWBalance")
    vect[22] =  feature_to_vector(entity, "NumBank2NatlTradesWHighUtilization")
    vect[23] =  feature_to_vector(entity, "PercentTradesWBalance")
    return vect
end


## bucketize for features
feature_ranges = Dict("ExternalRiskEstimate" => [64,71,76,81], 
                      "MSinceOldestTradeOpen" => [92,135,264],
                      "MSinceMostRecentTradeOpen"=> [20],
                      "AverageMInFile" => [49,70,97],
                      "NumSatisfactoryTrades" => [3,6,13,22],
                      "NumTrades60Ever2DerogPubRec" => [2,3,12,13],
                      "NumTrades90Ever2DerogPubRec" => [2,8,10],
                      "PercentTradesNeverDelq" => [59,84,89,96],
                      "MSinceMostRecentDelq" => [18,33,48],
                      "MaxDelq2PublicRecLast12M" => [6,7],
                      "MaxDelqEver" => [3],
                      "NumTotalTrades" => [1,10,17,28],
                      "NumTradesOpeninLast12M" => [3,4,7,12],
                      "PercentInstallTrades" => [36,47,58,85],
                      "MSinceMostRecentInqexcl7days" => [1,2,9,23],
                      "NumInqLast6M" => [2,5,9],
                      "NumInqLast6Mexcl7days" => [3],
                      "NetFractionRevolvingBurden" => [15,38,73],
                      "NetFractionInstallBurden" => [36,71],
                      "NumRevolvingTradesWBalance" => [4,5,8,12],
                      "NumInstallTradesWBalance" => [3,4,12,14],
                      "NumBank2NatlTradesWHighUtilization" => [2,3,4,6],
                      "PercentTradesWBalance" => [48,67,74,87])


# given entity and feature, produce the bucketized index for that pair
function feature_to_vector(entity, feature)
    ranges = feature_ranges[feature]
    val = entity[feature]
    if val < 0
        if val == -7
            return size(ranges)[1] + 2
        elseif val == -8
            return size(ranges)[1] + 3
        elseif val == -9
            return size(ranges)[1] + 4
        end
    else
        for (i, r) in enumerate(ranges)
            if val < r
                return i
            end
        end
    end
    return size(ranges)[1] + 1
end



## Subscore for each sublayer
function external_risk_subscore_from_vector(vect)
    weights = [2.9895622, 2.1651128, 1.4081029, 0.7686735, 0, 0, 0, 1.6943381] 
    score = -1.4308699
    return score + weights[vect[1]]
end

function trade_open_time_subscore_from_vector(vect)
    weights1 = [0.820842027, 0.525120503, 0.245257364, 0.005524848, 0, 0.418318111, 0.435851213] 
    weights2 = [0.031074792, 0.006016629, 0, 0, 0.027688067]
    weights3 = [ 1.209930852, 0.694452470, 0.296029824, 0, 0, 0, 0.471490736]
    score = -0.696619002
    return score + weights1[vect[2]] + weights2[vect[3]] + weights3[vect[4]]
end

function num_sat_trades_subscore_from_vector(vect)
    weights = [2.412574, 1.245278, 6.619963e-01, 2.731984e-01, 5.444148e-09, 0, 0, 4.338848e-01]
    score = -1.954726e-01
    return score + weights[vect[5]]
end

# high numbers!
function trade_freq_subscore_from_vector(vect)
    weights1 = [2.710260e-04, 9.195886e-01, 9.758620e-01, 1.008107e+01, 9.360290, 0, 0, 3.970360e-01] 
    weights2 = [1.514937e-01, 3.139667e-01, 0, 2.422345e-01, 0, 0, 3.095043e-02] 
    weights3 = [2.888436e-01, 9.659472e-01, 5.142479e-01, 2.653203e-01, 8.198233e-07, 0, 0, 3.233593e-01] 
    weights4 =[8.405069e-06, 3.374686e-01, 4.934466e-01, 8.601860e-01, 9.451724, 0, 0, 1.351433e-01]
    score = -6.480598e-01
    return score + weights1[vect[6]] + weights2[vect[7]] + weights3[vect[8]] + weights4[vect[9]]
end

function delinquency_subscore_from_vector(vect)
    weights1 = [1.658975, 1.218405, 8.030501e-01, 5.685712e-01, 0, 0, 0, 6.645698e-01] 
    weights2 = [4.014945e-01, 2.912651e-01, 5.665418e-02, 0,6.935965e-01, 5.470874e-01, 4.786956e-01]
    weights3 = [1.004642, 5.654694e-01, 0, 0, 0, 2.841047e-01]
    weights4 = [1.378803e-01, 1.101649e-06, 0, 0, 1.051132e-02]
    score = -1.199469
    return score + weights1[vect[10]] + weights2[vect[11]] + weights3[vect[12]] + weights4[vect[13]]
end

function installment_subscore_from_vector(vect)
    weights1 = [9.059412e-05, 1.292266e-01, 4.680034e-01, 8.117938e-01, 1.954441, 0, 0, 1.281830]
    weights2 = [0, 1.432068e-01, 3.705526e-01, 0, 4.972869e-03, 1.513885e-01]
    weights3 = [1.489759, 1.478176, 1.518328, 0, 9.585058e-01, 0, 1.506442, 5.561296e-01]
    score = -1.750937
    return score + weights1[vect[14]] + weights2[vect[15]] + weights3[vect[16]]
end

function inquiry_subscore_from_vector(vect)
    weights1 = [1.907737, 1.260966, 1.010585, 8.318137e-01, 0, 1.951357, 0, 1.719356]
    weights2 = [2.413596e-05, 2.251582e-01, 5.400251e-01, 1.255076, 0, 0, 1.061504e-01]
    weights3 = [0, 6.095516e-02, 0, 0, 1.125418e-02]
    score = -1.598351
    return score + weights1[vect[17]] + weights2[vect[18]] + weights3[vect[19]]
end

function revol_balance_subscore_from_vector(vect)
    weights1 = [0.0001042232, 0.6764476961, 1.3938464180, 2.2581926077, 0, 1.7708134303, 1.0411847907]
    weights2 = [0.0756555085, 0, 0.1175915408, 0.2823307493, 0.4242649887, 0, 0.8756715032, 0.0897134843]
    score = -0.8924856930
    return score + weights1[vect[20]] + weights2[vect[21]] 
end

function utilization_subscore_from_vector(vect)
    weights = [0, 0.8562096, 1.2047649, 1.1635459, 1.4701220, 0, 1.2392294, 0.4800086]
    score = -0.2415871
    return score + weights[vect[22]]
end

function trade_w_balance_subscore_from_vector(vect)
    weights = [0, 0.5966752, 0.9207121, 1.2749998, 1.8474869, 0, 2.2885183, 1.0606029]
    score = -0.8221922
    return score + weights[vect[23]]
end

# Sigmoid function
function sigmoid(x)
    return 1 / (1 + exp(-1 * x))
end


# the risk model to returen the risks for that given entity
function risk_model(entity)
    vect = entity_to_vectors(entity)
    subscore = Array{Float64,1}(undef, 10)
    subscore[1] = sigmoid(external_risk_subscore_from_vector(vect))
    subscore[2] = sigmoid(trade_open_time_subscore_from_vector(vect))
    subscore[3] = sigmoid(num_sat_trades_subscore_from_vector(vect))
    subscore[4] = sigmoid(trade_freq_subscore_from_vector(vect))
    subscore[5] = sigmoid(delinquency_subscore_from_vector(vect))
    subscore[6] = sigmoid(installment_subscore_from_vector(vect))
    subscore[7] = sigmoid(inquiry_subscore_from_vector(vect))
    subscore[8] = sigmoid(revol_balance_subscore_from_vector(vect))
    subscore[9] = sigmoid(utilization_subscore_from_vector(vect))
    subscore[10] = sigmoid(trade_w_balance_subscore_from_vector(vect))
    weights = [1.5671672, 2.5236825, 2.1711503, 0.3323177, 2.5396631, 0.9148520,
               3.0015073, 1.9259728, 0.9864329, 0.2949793]
    score = -8.3843046
    for i in 1:size(weights)[1]
        score += (subscore[i] * weights[i])
    end
    return sigmoid(score)
end

end ## End Module 
