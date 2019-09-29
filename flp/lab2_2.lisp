(defun odd_sum (ls)
    (if ls
        (+ (car ls) (odd_sum(cddr ls)))
        '0
    )
)

(princ (odd_sum '(-2 3 2 5 -6 5 2 1 3)))
