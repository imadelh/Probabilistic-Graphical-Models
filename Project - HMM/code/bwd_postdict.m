function [ lpp ] = bwd_postdict( A,log_p )

    mx = max(log_p(:)); 
    p = exp(log_p - mx);
    lpp = log(A'*p) + mx;

end

