kernel void ncc_stereo_matching(global const unsigned char* in_image_left, global const unsigned char* in_image_right, global char* out_disparity_left_to_right, int image_width, int image_height, int window_half_size, int max_disp) {
    int x = get_global_id(0);
    int y = get_global_id(1);


    //skip border region
    if (x < window_half_size || x >= image_width - window_half_size ||
        y < window_half_size || y >= image_height - window_half_size) {
        return;
    }

    //TODO: implement stereo matching

    float best_NCC_cost = -1.0; //initallize the matching cost 
    int best_disparity_hypothesis = 0;

    int e_limit_left_hand_side = window_half_size; //external limit
    int e_limit_right_hand_side = window_half_size;			//whyy warum nicht x + max_disp ?
    if (x > max_disp) {
        e_limit_left_hand_side = std::max(window_half_size, (x - max_disp));  //left border clamped
        e_limit_right_hand_side = std::min((image_width - 1) - window_half_size, x + max_disp); //right border clamped
    }

	for (int e = e_limit_left_hand_side; e <= e_limit_right_hand_side; ++e) {
		float current_NCC_cost = 0;
		float ref_window_sum = 0;
		float search_window_sum = 0;
		int counter_value = 0;
		float variance_ab = 0;
		float variance_a = 0;
		float variance_b = 0;

		// evaluate NCC cost between ref and search windows
		for (int win_y = -window_half_size; win_y <= window_half_size; ++win_y) {
			for (int win_x = -window_half_size; win_x <= window_half_size; ++win_x) {
				int search_index = ((y + win_y) * image_width + e + win_x);
				int reference_index = ((y + win_y) * image_width + x + win_x);
				ref_window_sum += in_image_left[reference_index];
				search_window_sum += in_image_right[search_index];

				variance_ab += in_image_left[reference_index] * in_image_right[search_index];
				variance_a += in_image_left[reference_index] * in_image_left[reference_index];
				variance_b += in_image_right[search_index] * in_image_right[search_index];

				++counter_value;
			}
		}

		float ref_window_mean = ref_window_sum / counter_value;
		float search_window_mean = search_window_sum / counter_value;
		variance_ab /= counter_value;
		variance_a /= counter_value;
		variance_b /= counter_value;

		//NCC
		current_NCC_cost = (variance_ab - ref_window_mean * search_window_mean)
			/ sqrt((variance_a - ref_window_mean * ref_window_mean) * (variance_b - search_window_mean * search_window_mean));

		// lower costs usually mean better match
		if (current_NCC_cost > best_NCC_cost) {
			best_NCC_cost = current_NCC_cost;
			/* - compute the current best disparity value
			 *      based on the postion of the current reference pixel from the left image
			 *      and the postion of the currently best matching serach pixel from the right image
			 */
			best_disparity_hypothesis = (x - e);
			//std::cout<< current_NCC_cost << '\n'; 

		}
	}

	int pixel_offset = (y * image_width + x);
	out_disparity_left_to_right[pixel_offset] = best_disparity_hypothesis; //write resulting output values

}