using PyPlot
using Gen

n_count_data = 80

alpha = 1.0 / 20.0

@gen function switch_point_model(n_count_data::Integer, alpha::Real)

    lambda_1 ~ exponential(alpha) # syntactic sugar for:
                                  # lambda_1 = {:lambda_1} ~ exponential(alpha)
    lambda_2 ~ exponential(alpha)
    tau ~ uniform_discrete(0, n_count_data)

    cnt_data = Integer[]

    for i = 1:n_count_data

        if tau < i
           push!(cnt_data, {(:cnt_data, i)} ~ poisson(lambda_1))
        else
           push!(cnt_data, {(:cnt_data, i)} ~ poisson(lambda_2))
        end

    end

    cnt_data

end

count_data = switch_point_model(n_count_data, alpha)

bar(1:n_count_data, count_data, color="#348ABD")
xlabel("Time (days)")
ylabel("count of text-msgs received")
title("Artificial dataset")
xlim(0, n_count_data)
plt[:show]()
