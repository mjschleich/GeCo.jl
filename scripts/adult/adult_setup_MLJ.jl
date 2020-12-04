using CSV, Statistics, DataFrames, MLJ

data = CSV.File("data/adult/adult_processed.csv") |> DataFrame
y, X = unpack(data, ==(:income), colname -> true);

tree = Nothing()

function learn_model()
    # load the model
    tree_model = @load RandomForestClassifier pkg=DecisionTree

    # change the input to the type they want
    Xs = coerce(X, 
        :race => OrderedFactor,
        :fnlwgt => Continuous,
        :capital_gain => Continuous,
        :occupation     => OrderedFactor,
        :relationship   => OrderedFactor,
        :sex            => OrderedFactor,
        :hours_per_week => Continuous,
        :capital_loss   => Continuous,
        :education_num  => Continuous,
        :native_country => OrderedFactor,
        :education      => OrderedFactor,
        :marital_status => OrderedFactor,
        :age            => Continuous,
        :workclass      => OrderedFactor
    )

    # change the target to the desired
    ys = categorical(data_y)

    models(matching(Xs, ys))
    global tree = machine(tree_model, Xs, ys)

    # split the dataset
    train, test = partition(eachindex(ys), 0.7, shuffle=true)

    # trainprint_tree(model, 5)
    fit!(tree, rows=train)
end

## it would be much more efficient to prodict some instead of one
# this is the risk_model that predict only one entity
function risk_model(entity)
    ec = DataFrame(entity)
    X = coerce(ec, :race => Multiclass,
        :fnlwgt => Continuous,
        :capital_gain => Continuous,
    :occupation     => OrderedFactor,
    :relationship   => OrderedFactor,
    :sex            => OrderedFactor,
    :hours_per_week => Continuous,
    :capital_loss   => Continuous,
    :education_num  => Continuous,
    :native_country => OrderedFactor,
    :education      => OrderedFactor,
    :marital_status => OrderedFactor,
    :age            => Continuous,
    :workclass      => OrderedFactor)
    yhat = predict(tree, X);
    return pdf(yhat[1], 1)
end

# this is the classify that predict only one entity
function classify(entity)
    ec = DataFrame(entity)
    X = coerce(ec, :race => Multiclass,
        :fnlwgt => Continuous,
        :capital_gain => Continuous,
    :occupation     => OrderedFactor,
    :relationship   => OrderedFactor,
    :sex            => OrderedFactor,
    :hours_per_week => Continuous,
    :capital_loss   => Continuous,
    :education_num  => Continuous,
    :native_country => OrderedFactor,
    :education      => OrderedFactor,
    :marital_status => OrderedFactor,
    :age            => Continuous,
    :workclass      => OrderedFactor)
    yhat = predict(tree, X);
    if pdf(yhat[1], 1) > 0.5
        return 1
    else
        return 0
    end
end

# this is the risk model for some entities (a DataFrame)
# return an array of the risk
function risk_models(entities)
    X = coerce(entities, :race => Multiclass,
        :fnlwgt => Continuous,
        :capital_gain => Continuous,
    :occupation     => OrderedFactor,
    :relationship   => OrderedFactor,
    :sex            => OrderedFactor,
    :hours_per_week => Continuous,
    :capital_loss   => Continuous,
    :education_num  => Continuous,
    :native_country => OrderedFactor,
    :education      => OrderedFactor,
    :marital_status => OrderedFactor,
    :age            => Continuous,
    :workclass      => OrderedFactor)
    yhat = predict(tree, X);
    return broadcast(pdf, yhat, 1)
end

# this is the classifier for some entities (a DataFrame)
# return an array of the classify
function classifies(entities)
    X = coerce(entities, :race => Multiclass,
        :fnlwgt => Continuous,
        :capital_gain => Continuous,
    :occupation     => OrderedFactor,
    :relationship   => OrderedFactor,
    :sex            => OrderedFactor,
    :hours_per_week => Continuous,
    :capital_loss   => Continuous,
    :education_num  => Continuous,
    :native_country => OrderedFactor,
    :education      => OrderedFactor,
    :marital_status => OrderedFactor,
    :age            => Continuous,
    :workclass      => OrderedFactor)
    yhat = predict(tree, X);
    y_class = broadcast(pdf, yhat, 1)
    for i in size(y_class)[0]
        y_class[i] = y_class[i] >= 0.5 ? 1 : 0
    end
    return y_class
end


