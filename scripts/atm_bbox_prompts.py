"""Local copy of the ATM bbox prompt dictionary used for LIBERO tasks."""

from __future__ import annotations


def get_task_to_prompt_dict() -> dict[str, dict[str, str]]:
    return {
        "libero_goal": {
            "open_the_middle_drawer_of_the_cabinet_demo": "the black drawer.",
            "open_the_top_drawer_and_put_the_bowl_inside_demo": "the black drawer. the gray bowl.",
            "push_the_plate_to_the_front_of_the_stove_demo": "the pink-white plate. the gas stove.",
            "put_the_bowl_on_the_plate_demo": "the pink-white plate. the gray bowl.",
            "put_the_bowl_on_the_stove_demo": "the gas stove. the gray bowl.",
            "put_the_bowl_on_top_of_the_cabinet_demo": "the black drawer. the gray bowl.",
            "put_the_cream_cheese_in_the_bowl_demo": "the suger box. the gray bowl.",
            "put_the_wine_bottle_on_the_rack_demo": "the bottle. the stripped rack.",
            "put_the_wine_bottle_on_top_of_the_cabinet_demo": "the bottle. the black drawer.",
            "turn_on_the_stove_demo": "the gas stove.",
        },
        "libero_spatial": {
            "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo": (
                "the gray bowl. the pink-white plate."
            ),
            "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo": (
                "the gray bowl. the pink-white plate."
            ),
            "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo": (
                "the gray bowl. the pink-white plate."
            ),
            "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo": (
                "the gray bowl. the pink-white plate."
            ),
            "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo": (
                "the gray bowl. the pink-white plate."
            ),
            "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo": (
                "the gray bowl. the pink-white plate."
            ),
            "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo": (
                "the gray bowl. the pink-white plate."
            ),
            "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo": (
                "the gray bowl. the pink-white plate."
            ),
            "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo": (
                "the gray bowl. the pink-white plate."
            ),
            "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo": (
                "the gray bowl. the pink-white plate."
            ),
        },
        "libero_object": {
            "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo": "the basket.",
            "pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo": "the basket.",
            "pick_up_the_butter_and_place_it_in_the_basket_demo": "the basket.",
            "pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo": "the basket.",
            "pick_up_the_cream_cheese_and_place_it_in_the_basket_demo": "the basket.",
            "pick_up_the_ketchup_and_place_it_in_the_basket_demo": "the basket.",
            "pick_up_the_milk_and_place_it_in_the_basket_demo": "the basket.",
            "pick_up_the_orange_juice_and_place_it_in_the_basket_demo": "the basket.",
            "pick_up_the_salad_dressing_and_place_it_in_the_basket_demo": "the basket.",
            "pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo": "the basket.",
        },
        "libero_10": {
            "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo": "the moka pot.",
            "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo": (
                "the black drawer. the gray bowl."
            ),
            "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo": (
                "the yellow cup."
            ),
            "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo": "the moka pot.",
            "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo": (
                "the basket."
            ),
            "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo": (
                "the basket."
            ),
            "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo": (
                "the basket."
            ),
            "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo": (
                "the pink-white plate."
            ),
            "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo": (
                "the pink-white plate."
            ),
            "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo": (
                "the book. the caddy."
            ),
        },
        "real_atm_qpos": {
            "put_the_banana_into_the_basket": "the banana. the basket.",
            "put_the_carrot_into_the_basket": "the carrot. the basket.",
            "put_the_sponge_onto_the_plate": "the sponge. the plate.",
        },
        "real_atm_qpos_45_25": {
            "put_the_banana_into_the_basket": "the banana. the basket.",
            "put_the_carrot_into_the_basket": "the carrot. the basket.",
            "put_the_sponge_onto_the_plate": "the sponge. the plate.",
        },
        "real_atm_qpos_45_50": {
            "put_the_banana_into_the_basket": "the banana. the basket.",
            "put_the_carrot_into_the_basket": "the carrot. the basket.",
            "put_the_sponge_onto_the_plate": "the sponge. the plate.",
        },
        "real_atm_qpos_90_25": {
            "put_the_banana_into_the_basket": "the banana. the basket.",
            "put_the_carrot_into_the_basket": "the carrot. the basket.",
            "put_the_sponge_onto_the_plate": "the sponge. the plate.",
        },
        "real_atm_qpos_90_50": {
            "put_the_banana_into_the_basket": "the banana. the basket.",
            "put_the_carrot_into_the_basket": "the carrot. the basket.",
            "put_the_sponge_onto_the_plate": "the sponge. the plate.",
        },
        "atm_0828_spatial": {
            "place_the_lemon_lower_left_of_the_red_plate_onto_the_plate": (
                "a small yellow non-blue lemon. the red plate."
            ),
            "place_the_lemon_lower_right_of_the_red_plate_onto_the_plate": (
                "a small yellow non-blue lemon. the red plate."
            ),
            "place_the_lemon_upper_left_of_the_red_plate_onto_the_plate": (
                "a small yellow non-blue lemon. the red plate."
            ),
            "place_the_lemon_upper_right_of_the_red_plate_on_the_plate": (
                "a small yellow non-blue lemon. the red plate."
            ),
            "take_out_a_tissue_from_the_tissue_box_on_the_plate": "the blue-white tissue box. the red plate.",
            "take_out_a_tissue_from_the_tissue_box_on_the_table": "the blue-white tissue box.",
        },
        "h2r_robot_0826": {
            "pick_up_a_pot_and_place_it_onto_the_stove": "black pot. the brown-black non-blue square toy.",
            "pick_up_the_broom_and_sweep_the_toy_onto_the_dustpan": (
                "the brush. the orange non-blue toy. the white dustpan"
            ),
        },
    }
