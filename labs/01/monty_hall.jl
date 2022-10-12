using Statistics
using Gen

@gen function monty_hall_model()

   car_door = {:car_door} ~ uniform_discrete(1, 3)
   picked_door = {:picked_door} ~ uniform_discrete(1, 3)
   preference = {:preference} ~ uniform_discrete(0, 1)

   function host_choice()
      
      if car_door != picked_door
         return 6 - car_door - picked_door
      end
      if car_door == 1
         left = 2
         right = 3
      else
         left = 1
         if car_door == 2
            right = 3
         else
            right = 2
         end
      end
      return preference == 1 ? right : left

   end

   function changed_door()

      return 6 - host_choice() - picked_door

   end

   return changed_door()

end

traces = [Gen.simulate(monty_hall_model, ()) for _=1:40000];

# println(traces[1][:car_door])
# println(traces[1][:picked_door])
# println(traces[1][])

print("probability to win of a player who stays with the initial choice: ")
println(mean([traces[i][:car_door] == traces[i][:picked_door] for i=1:40000]))
print("probability to win of a player who switches: ")
println(mean([traces[i][:car_door] == traces[i][] for i=1:40000]))


