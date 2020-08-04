(define (domain cartpole-initial-bhaskara)

  (:requirements :typing :durative-actions :disjunctive-preconditions :duration-inequalities :fluents :time :negative-preconditions :timed-initial-literals )

  (:types cart )

  (:predicates
    (total_failure) (pole_position) (ready)
  )

  (:functions
    (x) (x_dot) (x_ddot) (m_cart) (friction_cart) 
    (theta) (theta_dot) (theta_ddot)
    (l_pole) (m_pole) (friction_pole)
    (gravity) (F) (elapsed_time) (inertia)
  )

  (:process movement
    :parameters ()
    :precondition (and (ready))
    :effect (and 
        (increase (x_ddot) (* #t
                                (/ 
                                    (- 
                                      (* (+ (inertia) (* (m_pole) (* (l_pole) (l_pole)))) (+ (F) (* (m_pole) (* (l_pole) (* (* (theta_dot) (theta_dot)) (/ (* (* 4 (* (theta) 57.29578)) (- 180 (* (theta) 57.29578))) (- 40500 (* (* (theta) 57.29578) (- 180 (* (theta) 57.29578))))) )))) ) 
                                      (* (gravity) (* (* (* (m_pole) (m_pole)) (* (l_pole) (l_pole)) ) (* (/ (* (* 4 (* (theta) 57.29578)) (- 180 (* (theta) 57.29578))) (- 40500 (* (* (theta) 57.29578) (- 180 (* (theta) 57.29578))))) (/ (- 32400 (* 4 (* (* (theta) 57.29578) (* (theta) 57.29578)))) (+ 32400 (* (* (theta) 57.29578) (* (theta) 57.29578)))) ) ) ) 
                                    )

                                    (+ 
                                        (* (inertia) (+ (m_cart) (m_pole)) ) 
                                        (* (* (m_pole) (* (l_pole) (l_pole))) (+ (m_cart) (* (m_pole) (* (/ (* (* 4 (* (theta) 57.29578)) (- 180 (* (theta) 57.29578))) (- 40500 (* (* (theta) 57.29578) (- 180 (* (theta) 57.29578))))) (/ (* (* 4 (* (theta) 57.29578)) (- 180 (* (theta) 57.29578))) (- 40500 (* (* (theta) 57.29578) (- 180 (* (theta) 57.29578))))) ) ) ) ) 
                                    ) 
                                ) 
                            )
        )

        (increase (theta_ddot) (* #t 
                                    (* 
                                        -1 
                                        (/ 
                                            (* 
                                                (* (m_pole) (l_pole)) 
                                                (- (+ (* (F) (/ (- 32400 (* 4 (* (* (theta) 57.29578) (* (theta) 57.29578)))) (+ 32400 (* (* (theta) 57.29578) (* (theta) 57.29578))))) (* (m_pole) (* (l_pole) (* (* (theta_dot) (theta_dot)) (* (/ (* (* 4 (* (theta) 57.29578)) (- 180 (* (theta) 57.29578))) (- 40500 (* (* (theta) 57.29578) (- 180 (* (theta) 57.29578))))) (/ (- 32400 (* 4 (* (* (theta) 57.29578) (* (theta) 57.29578)))) (+ 32400 (* (* (theta) 57.29578) (* (theta) 57.29578))))) ) ) ) ) (* (+ (m_cart) (m_pole)) (* (gravity) (/ (* (* 4 (* (theta) 57.29578)) (- 180 (* (theta) 57.29578))) (- 40500 (* (* (theta) 57.29578) (- 180 (* (theta) 57.29578)))))) ) ) 
                                            )
                                            (+ 
                                                (* (inertia) (+ (m_cart) (m_pole)) ) 
                                                (* (* (m_pole) (* (l_pole) (l_pole))) (+ (m_cart) (* (m_pole) (* (/ (* (* 4 (* (theta) 57.29578)) (- 180 (* (theta) 57.29578))) (- 40500 (* (* (theta) 57.29578) (- 180 (* (theta) 57.29578))))) (/ (* (* 4 (* (theta) 57.29578)) (- 180 (* (theta) 57.29578))) (- 40500 (* (* (theta) 57.29578) (- 180 (* (theta) 57.29578)))))) ) ) ) 
                                            ) 
                                        ) 
                                    )   
                                )
        )

        (increase (x_dot) (* #t (x_ddot)) )
        (increase (theta_dot) (* #t (theta_ddot)) )
        (increase (x) (* #t (x_dot)) )
        (increase (theta) (* #t (theta_dot)))
        (increase (elapsed_time) (* #t 1) )
    )
  )

  (:durative-action move_cart_right
    :parameters ()
    :duration (= ?duration 1)
    :condition (and (over all (ready)))
    :effect (and 
      (at start (assign (F) 1)) 
      (at end (assign (F) 0))
  ) 
  )
  
  (:durative-action move_cart_left
    :parameters ()
    :duration (= ?duration 1)
    :condition (and (over all (ready)))
    :effect (and 
      (at start (assign (F) -1)) 
      (at end (assign (F) 0))
  ) 
  )

  (:event entered_goal_region
      :parameters ()
      :precondition (and
          (<= (theta) 0.2618)
          (>= (theta) -0.2618)
          (not(pole_position))
      )
      :effect (and
          (pole_position)
      )
  )

  (:event exited_goal_region
      :parameters ()
      :precondition (and
          (>= (theta) 0.2618)
          (<= (theta) -0.2618)
          (pole_position)
      )
      :effect (and
          (total_failure)
      )
  )
  

)
