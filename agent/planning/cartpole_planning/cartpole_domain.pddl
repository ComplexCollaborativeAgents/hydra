(define (domain cartpole-initial-bhaskara)

  (:requirements :typing :fluents :time :negative-preconditions)

  (:types dummy )

  (:predicates
    (total_failure) (ready) (cart_available)
  )

  (:functions
    (x) (x_dot) (x_ddot) (m_cart) (friction_cart)
    (theta) (theta_dot) (theta_ddot)
    (l_pole) (m_pole) (friction_pole)
    (gravity) (F) (elapsed_time) (inertia)
    (time_limit) (angle_limit) (x_limit) (force_mag)
  )


  ; TEMP =  	(/ (+ (F) (* (* (m_pole) (l_pole)) (* (* (theta_dot) (theta_dot)) (SIN_THETA) ) ) ) (+ (m_cart) (m_pole)) )
  ; COS_THETA = (/ (- 32400 (* 4 (* (* (theta) 57.29578) (* (theta) 57.29578)))) (+ 32400 (* (* (theta) 57.29578) (* (theta) 57.29578))))
  ; SIN_THETA = (/ (* (* 4 (* (theta) 57.29578)) (- 180 (* (theta) 57.29578))) (- 40500 (* (* (theta) 57.29578) (- 180 (* (theta) 57.29578)))))


  (:event set_force
      :parameters (?d - dummy)
      :precondition (and (ready) (not (total_failure)))
      :effect (and
      	(cart_available)
      	(assign (theta_ddot)
      		(/
      			(- (* (gravity) (/ (* (* 4 (* (theta) 57.29578)) (- 180 (* (theta) 57.29578))) (- 40500 (* (* (theta) 57.29578) (- 180 (* (theta) 57.29578))))) ) (* (/ (- 32400 (* 4 (* (* (theta) 57.29578) (* (theta) 57.29578)))) (+ 32400 (* (* (theta) 57.29578) (* (theta) 57.29578)))) (/ (+ (F) (* (* (m_pole) (l_pole)) (* (* (theta_dot) (theta_dot)) (/ (* (* 4 (* (theta) 57.29578)) (- 180 (* (theta) 57.29578))) (- 40500 (* (* (theta) 57.29578) (- 180 (* (theta) 57.29578))))) ) ) ) (+ (m_cart) (m_pole)) ) ) )
      			(* (l_pole) (- (/ 4.0 3.0) (/ (* (m_pole) (* (/ (- 32400 (* 4 (* (* (theta) 57.29578) (* (theta) 57.29578)))) (+ 32400 (* (* (theta) 57.29578) (* (theta) 57.29578)))) (/ (- 32400 (* 4 (* (* (theta) 57.29578) (* (theta) 57.29578)))) (+ 32400 (* (* (theta) 57.29578) (* (theta) 57.29578)))) )) (+ (m_cart) (m_pole)) ) ) )
  			)
      	)
      	(assign (x_ddot)
      		(-
      			(/ (+ (F) (* (* (m_pole) (l_pole)) (* (* (theta_dot) (theta_dot)) (/ (* (* 4 (* (theta) 57.29578)) (- 180 (* (theta) 57.29578))) (- 40500 (* (* (theta) 57.29578) (- 180 (* (theta) 57.29578))))) ) ) ) (+ (m_cart) (m_pole)) )
      			(/ (* (* (m_pole) (l_pole)) (* (theta_ddot) (/ (- 32400 (* 4 (* (* (theta) 57.29578) (* (theta) 57.29578)))) (+ 32400 (* (* (theta) 57.29578) (* (theta) 57.29578)))) )) (+ (m_cart) (m_pole)) )
  			)
      	)
      )
  )

  (:process movement
    :parameters (?d - dummy)
    :precondition (and (ready) (not (total_failure)))
    :effect (and
        (increase (x) (* #t (x_dot)) )
        (increase (theta) (* #t (theta_dot)))
        (increase (x_dot) (* #t (x_ddot)) )
        (increase (theta_dot) (* #t (theta_ddot)) )
        (increase (elapsed_time) (* #t 1) )
    )
  )


  (:action move_cart_left
    :parameters (?d - dummy)
    :precondition (and
    	(ready)
    	(= (F) (force_mag))
    	(cart_available)
    	(not (total_failure))
	)
    :effect (and
      (assign (F) (- 0.0 (force_mag)))
      (not (cart_available))
  	)
  )

  (:action move_cart_right
    :parameters (?d - dummy)
    :precondition (and
    	(ready)
    	(= (F) (- 0.0 (force_mag)))
    	(cart_available)
    	(not (total_failure))
	)
    :effect (and
      (assign (F) (force_mag))
      (not (cart_available))
  	)
  )

  ; (:event entered_goal_region
  ;     :parameters (?d - dummy)
  ;     :precondition (and
  ;         (<= (theta) (angle_limit))
  ;         (>= (theta) (- 0.0 (angle_limit)) )
  ;         (not (pole_position))
  ;         (not (total_failure))
  ;     )
  ;     :effect (and
  ;         (pole_position)
  ;     )
  ; )

  (:event exited_goal_region
      :parameters (?d - dummy)
      :precondition (and
          (or (>= (theta) (angle_limit))
          (<= (theta) (- 0.0 (angle_limit))) )
          ; (pole_position)
          (not (total_failure))
      )
      :effect (and
      	  ; (not (pole_position))
          (total_failure)
      )
  )

  (:event cart_out_of_bounds
      :parameters (?d - dummy)
      :precondition (and
          (or (>= (x) (x_limit))
          (<= (x) (- 0.0 (x_limit))) )
          (not (total_failure))
      )
      :effect (and
          (total_failure)
      )
  )

  (:event time_limit_reached
      :parameters (?d - dummy)
      :precondition (and
          (> (elapsed_time) (time_limit))
          (not (total_failure))
      )
      :effect (and
          (total_failure)
      )
  )


)
