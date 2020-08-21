function [q, steps_per_episode] = qlearning(episodes)

% set up parameters and initialize q values
alpha = 0.05;
gamma = 0.99;
num_states = 100;
num_actions = 2;
actions = [-1, 1];
q = zeros(num_states, num_actions);
steps_per_episode = zeros(episodes,1);

% Take trials equal to the number of episodes passed in as the parameter
for i=1:episodes
  steps = 0;
  % first retrieve the initialized values of the car by passing in an
  % action of "0"
  [x, s, absorb] =  mountain_car([0.0 -pi/6], 0);
  % next retrieve maximum next action at the current state
  [~, ai] = max(q(s,:));
  % if there is no maximum, choose a random action 
  if (q(s, 1) == q(s, 2))
      ai = ceil(rand * num_actions);
  end
  
  %while the car has not entered an absorbing state
  while (absorb == 0)
    % move the car accord to action ai
    [x, sprime, absorb] = mountain_car(x, actions(ai));
    % set the reward function is -1 if the car is in an absorbing state
    R = -double(absorb == 0);
    % copmute the maximum action to take in the next step
    [~, ai_prime] = max(q(sprime,:));
    % randomize that action if there is no maximum
    if (q(sprime, 1) == q(sprime, 2))
        ai_prime = ceil(rand * num_actions); 
    end
    % update Q for the current state
    q(s, ai) = (1-alpha)*q(s, ai) + alpha * (R + gamma * q(sprime, ai_prime));
    % set ai and s to ai_prime and s_prime so that in the next iteration
    % the q matrix is updated according to the updated state and action
    ai = ai_prime;
    s = sprime;
    
    %increase the counter of the step at this episode
    steps = steps + 1;
  end
  
  %set the steps of episode in the current iteration
  steps_per_episode(i) = steps;
end