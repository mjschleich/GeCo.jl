## This file contians the functions that encode possible summary representations of the generated explanations

using Printf

function actions(counterfactuals::DataFrame, orig_instance::DataFrameRow;
    num_actions::Int64=5, output::String="text")
    out = ""
    for idx in 1:min(num_actions, nrow(counterfactuals))
        cf = counterfactuals[idx,:]
        if output == "text"
            out *= "COUNTERFACTUAL $(idx)\\n Desired Outcome: $(cf.outc),\\t Score: $(cf.score)\\n "
        elseif output == "md"
            out *= "\\\n**COUNTERFACTUAL $(idx)**\\\nDesired Outcome: $(cf.outc),\tScore: $(cf.score)\\\n"
        elseif output == "html"
            out *= "<b>COUNTERFACTUAL $(idx)</b>\nDesired Outcome: $(cf.outc),\tScore: $(cf.score)\n"
        end
        for feature in propertynames(orig_instance)
            if cf[feature] != orig_instance[feature]
                if output == "text"
                    out *= "$feature : \t$(orig_instance[feature]) --> $(cf[feature])\\n"
                elseif output == "md"
                    out*= "$feature : \t$(orig_instance[feature]) --> $(cf[feature])\\\n"
                elseif output == "html"
                    out *= "$feature : \t$(orig_instance[feature]) \$\\to\$ $(cf[feature])\\\n"
                end
            end
        end
        out *= "*********\\\n"
    end

    out
end


function actions(counterfactuals::DataManager, orig_instance; num_actions = 5, output::String="text")
    # Turn DataManager into a DataFrame
    df = materialize(counterfactuals)
    DataFrame.sort!(df, :score)
    actions(df, orig_instance; num_actions=num_actions, output=output)
end

function actionsDemo(counterfactuals::DataFrame, orig_instance::DataFrameRow)
    outarray = Vector{String}()
    for idx in 1:nrow(counterfactuals)
        cf = counterfactuals[idx,:]
        out = "**COUNTERFACTUAL $(idx)**\\\n"
        for feature in propertynames(orig_instance)
            if cf[feature] != orig_instance[feature]
                out*= "$feature : \t$(orig_instance[feature]) \$\\to\$ $(cf[feature])\\\n"
            end
        end
        out *= "Score: $(@sprintf("%.5f", cf.score))\\\n"
        push!(outarray, out)
    end
    outarray
end
