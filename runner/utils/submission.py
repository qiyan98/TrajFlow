import os
import tarfile
from waymo_open_dataset.protos.motion_submission_pb2 import *


def traj_serialize(trajectories, scores, object_ids):
    scored_obj_trajs = []
    for i in range(trajectories.shape[0]):
        center_x, center_y = trajectories[i, 4::5, 0], trajectories[i, 4::5, 1]
        traj = Trajectory(center_x=center_x, center_y=center_y)
        object_traj = ScoredTrajectory(confidence=scores[i], trajectory=traj)
        scored_obj_trajs.append(object_traj)
    return SingleObjectPrediction(trajectories=scored_obj_trajs, object_id=object_ids)


def serialize_single_scenario(scenario_list):
    single_prediction_list = []
    scenario_id = scenario_list[0]['scenario_id']
    # Assert all scenario_ids match once
    assert all(d['scenario_id'] == scenario_id for d in scenario_list)
    for single_dict in scenario_list:
        sc_id = single_dict['scenario_id']
        single_prediction = traj_serialize(single_dict['pred_trajs'],
            single_dict['pred_scores'], single_dict['object_id'])
        single_prediction_list.append(single_prediction)
    prediction_set = PredictionSet(predictions=single_prediction_list)
    return ChallengeScenarioPredictions(scenario_id=scenario_id, single_predictions=prediction_set)


def joint_serialize_single_scenario(scenario_list):
    assert len(scenario_list)==2
    scenario_id = scenario_list[0]['scenario_id']
    joint_score = scenario_list[0]['pred_scores']
    full_scored_trajs = []
    for j in range(6):
        object_trajs = []
        for i in range(2):
            center_x = scenario_list[i]['pred_trajs'][j, 4::5, 0]
            center_y = scenario_list[i]['pred_trajs'][j, 4::5, 1]
            traj = Trajectory(center_x=center_x, center_y=center_y)
            score_traj = ObjectTrajectory(object_id=scenario_list[i]['object_id'], trajectory=traj)
            object_trajs.append(score_traj)   
        full_scored_trajs.append(ScoredJointTrajectory(trajectories=object_trajs, confidence=joint_score[j]))
    joint_prediction = JointPrediction(joint_trajectories=full_scored_trajs)
    return ChallengeScenarioPredictions(scenario_id=scenario_id, joint_prediction=joint_prediction)


def serialize_single_batch(final_pred_dicts, joint_pred=False):
    ret_scenarios = []
    for b in range(len(final_pred_dicts)):
        scenario_list = final_pred_dicts[b]
        if joint_pred:
            scenario_preds = joint_serialize_single_scenario(scenario_list)
        else:
            scenario_preds = serialize_single_scenario(scenario_list)
        ret_scenarios.append(scenario_preds)
    return ret_scenarios


def save_submission_file(scenerio_predictions, inter_pred, save_dir, save_name, submission_info, logger):
    submission_type = 2 if inter_pred else 1

    submission = MotionChallengeSubmission(
        account_name=submission_info['account_name'], 
        unique_method_name=submission_info['unique_method_name'],
        authors=submission_info['authors'], 
        affiliation=submission_info['affiliation'], 
        submission_type=submission_type, 
        scenario_predictions=scenerio_predictions,
        uses_lidar_data=submission_info['uses_lidar_data'],
        uses_camera_data=submission_info['uses_camera_data'],
        uses_public_model_pretraining=submission_info['uses_public_model_pretraining'],
        public_model_names=submission_info['public_model_names'],
        num_model_parameters=submission_info['num_model_parameters']
        )

    os.makedirs(save_dir, exist_ok=True)
    proto_path = os.path.join(save_dir, f"{save_name}.proto")
    tar_path = os.path.join(save_dir, f"{save_name}.gz")

    with open(proto_path, "wb") as f:
        f.write(submission.SerializeToString())

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(proto_path)

    os.remove(proto_path)

    logger.info("Submission file saved to {:s} with {:d} trajectories...".format(tar_path, len(scenerio_predictions)))
