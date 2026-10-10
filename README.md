function [fbest,sbest,fbest_history] = IDOA(pop,T,lb,ub,D,fobj)
if isscalar(lb), lb_vec = lb.*ones(1,D); else, lb_vec = lb; end
if isscalar(ub), ub_vec = ub.*ones(1,D); else, ub_vec = ub; end
x = initialization(pop, D, ub_vec, lb_vec);
fitness = inf(pop,1);
fbest = inf;
sbest = ones(1,D);
for j = 1:pop
    fitness(j) = fobj(x(j,:));
    if fitness(j) < fbest
        fbest = fitness(j);
        sbest = x(j,:);
    end
end
nGroups = 5;
sbestd = repmat(sbest, nGroups, 1);
fbestd = repmat(fbest, nGroups, 1);
fbest_history = zeros(1, T);
levy_scale_factor_initial = 0.2;
levy_scale_factor_final   = 0.005;
beta_levy_initial = 1.2;
beta_levy_final   = 1.8;
memory_prob_initial = 0.9;
memory_prob_final   = 0.2;
elite_learn_initial = 0.6;
elite_learn_final   = 0.1;
T1 = floor(0.9*T);
for i_iter = 1:T1
    current_levy_scale  = levy_scale_factor_initial - (levy_scale_factor_initial - levy_scale_factor_final) * (i_iter / T);
    current_beta_levy   = beta_levy_initial + (beta_levy_final - beta_levy_initial) * (i_iter / T);
    current_memory_prob = memory_prob_initial - (memory_prob_initial - memory_prob_final) * (i_iter / T);
    current_elite_learn = elite_learn_initial - (elite_learn_initial - elite_learn_final) * (i_iter / T);
    beta_perturb = 1 - current_elite_learn;
    for m = 1:nGroups
        idx1 = floor((m-1)*pop/nGroups) + 1;
        idx2 = floor(m*pop/nGroups);
        idx  = idx1:idx2;
        if isempty(idx), continue; end
        [fmin, pmin] = min(fitness(idx));
        fbestd(m) = fmin;
        sbestd(m,:) = x(idx(pmin),:);
        k = randi([ceil(D/(8*m)), max(2, ceil(D/(3*m)))]);
        k = min(k, D);
        for j = idx
            cand = x(j,:);
            if rand() < current_memory_prob
                cand = sbestd(m,:);
            end
            in = randperm(D, k);
            for h = 1:k
                levy_step = levy_flight_step_scalar(current_beta_levy);
                perturb = levy_step * current_levy_scale * (ub_vec(in(h)) - lb_vec(in(h)));
                cand(in(h)) = cand(in(h)) + perturb;
                cand(in(h)) = max(lb_vec(in(h)), min(ub_vec(in(h)), cand(in(h))));
            end
            cand(in) = cand(in) + current_elite_learn * (sbest(in) - cand(in)) + beta_perturb * rand(1, length(in));
            cand(in) = max(lb_vec(in), min(ub_vec(in), cand(in)));
            candF = fobj(cand);
            if candF < fitness(j)
                x(j,:) = cand;
                fitness(j) = candF;
            end
            if fitness(j) < fbestd(m)
                fbestd(m) = fitness(j);
                sbestd(m,:) = x(j,:);
            end
            if fitness(j) < fbest
                fbest = fitness(j);
                sbest = x(j,:);
            end
        end
    end
    if i_iter == 1
        fbest_history(i_iter) = fbest;
    else
        fbest_history(i_iter) = min(fbest_history(i_iter-1), fbest);
    end
end
for i_iter = T1+1:T
    current_levy_scale  = levy_scale_factor_initial - (levy_scale_factor_initial - levy_scale_factor_final) * (i_iter / T);
    current_beta_levy   = beta_levy_initial + (beta_levy_final - beta_levy_initial) * (i_iter / T);
    current_memory_prob = memory_prob_initial - (memory_prob_initial - memory_prob_final) * (i_iter / T);
    current_elite_learn = elite_learn_initial - (elite_learn_initial - elite_learn_final) * (i_iter / T);
    beta_perturb = 1 - current_elite_learn;
    for j = 1:pop
        cand = x(j,:);
        if D >= 2
            km = max(2, ceil(D/3));
            km = min(km, D);
            k = randi([2, km]);
        else
            k = 1;
        end
        in = randperm(D, k);
        if rand() < current_memory_prob
            cand = sbest;
        end
        for h = 1:k
            levy_step = levy_flight_step_scalar(current_beta_levy);
            perturb = levy_step * current_levy_scale * (ub_vec(in(h)) - lb_vec(in(h)));
            cand(in(h)) = cand(in(h)) + perturb;
            cand(in(h)) = max(lb_vec(in(h)), min(ub_vec(in(h)), cand(in(h))));
        end
        cand(in) = cand(in) + current_elite_learn * (sbest(in) - cand(in)) + beta_perturb * rand(1, length(in));
        cand(in) = max(lb_vec(in), min(ub_vec(in), cand(in)));
        candF = fobj(cand);
        if candF < fitness(j)
            x(j,:) = cand;
            fitness(j) = candF;
        end
        if fitness(j) < fbest
            fbest = fitness(j);
            sbest = x(j,:);
        end
    end
    if i_iter == 1
        fbest_history(i_iter) = fbest;
    else
        fbest_history(i_iter) = min(fbest_history(i_iter-1), fbest);
    end
end
end

function x = initialization(pop,dim,ub,lb)
Boundary_no = size(ub,2);
if Boundary_no == 1
    x = rand(pop,dim).*(ub-lb) + lb;
else
    x = zeros(pop,dim);
    for i = 1:dim
        x(:,i) = rand(pop,1).*(ub(i)-lb(i)) + lb(i);
    end
end
end

function step = levy_flight_step_scalar(beta_levy)
if nargin < 1, beta_levy = 1.5; end
sigma = (gamma(1+beta_levy)*sin(pi*beta_levy/2) / ...
    (gamma((1+beta_levy)/2)*beta_levy*2^((beta_levy-1)/2)))^(1/beta_levy);
u = randn() * sigma;
v = randn();
step = u / (abs(v)^(1/beta_levy));
end
