# genetic-counterfactual


To run:

```Julia
using Pkg; Pkg.activate(".")
using Revise, GeneticCounterfactual
```

Load one of the following setup script:
```Julia
include("scripts/fico/fico-setup.jl");
include("scripts/credit/credit_setup.jl");
include("scripts/credit/credit_setup_MACE.jl");
include("scripts/credit/credit_setup_PRF.jl");
include("scripts/adult/adult_setup_MACE.jl");
include("scripts/adult/adult_setup_Dice.jl");
include("scripts/tpcds/tpcds-setup.jl");
include("scripts/allstate/allstate_setup_PRF.jl");
include("scripts/yelp/yelp_setup_PRF.jl");
```

To compute the explanation:
```Julia
explanation = @time explain(orig_entity, X, path, classifier; desired_class = 1)
```

To print out actions:
```Julia
actions(explanation, orig_entity)
```

To precompute the domains run:
```Julia
domains = initDomains(path, X)

explanation, = @time explain(orig_entity, X, path, classifier; desired_class = 1, domains=domains, verbose=true)
```