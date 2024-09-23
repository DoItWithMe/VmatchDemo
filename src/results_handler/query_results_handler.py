from typing import Any
from loguru import logger as log
from pydantic import BaseModel
from pymilvus import MilvusClient
from utils.time_utils import milliseconds_to_hhmmss


class MediaDes(BaseModel):
    name: str = ""
    frame_total_len: int = -1
    avg_frame_time: float = 125.0


class QueryResult(BaseModel):
    ref_start_index: int = -1
    ref_end_index: int = -1
    sample_start_index: int = -1
    sample_end_index: int = -1
    remove_flag: bool = False
    distance: float = 0


class QueryResultHandler:
    def __init__(self, segment_len_limit: int):
        self.query_result_dict: dict[str, list[QueryResult]] = dict()
        self.quick_search_result_dict: dict[str, list[int]] = dict()

        self.segment_len_limit: int = segment_len_limit
        self.print_flag: bool = False
        self.merge_total_cost: float = 0.0
        self.candidate_cost: float = 0.0
        self.pare_cost: float = 0.0
        self.sort_cost: float = 0.0

        self.merge_query_res_inner_cost = 0.0

    @classmethod
    def _could_merge(cls, s, e, l, u) -> bool:
        if s <= l and e >= l:
            return True

        if s >= l and s <= u:
            return True

        return False

    @classmethod
    def _merge_distance(
        cls, s1: int, e1: int, s2: int, e2: int, dis1: float, dis2: float
    ) -> float:
        len1 = e1 - s1
        len2 = e2 - s2
        if len1 == 0 or len2 == 0:
            return dis1 + dis2

        avg_dis1 = dis1 / len1
        avg_dis2 = dis2 / len2

        if e1 <= s2:
            return dis1 + dis2
        elif s1 < s2 and e1 <= e2:
            # return avg_dis1 * (e1 / 2 - s1 + s2 / 2) + avg_dis2 * (
            #     e2 - e1 / 2 - s2 / 2
            # )
            return dis1 + avg_dis2 * (e2 - e1)
        elif s1 < s2 and e1 > e2:
            # return dis2 / 2 + avg_dis1 * (len1 - len2 / 2)
            return dis1
        elif s1 >= s2 and e1 <= e2:
            # return dis1 / 2 + avg_dis2 * (len2 - len1 / 2)
            return dis2
        elif s1 >= s2 and e1 > e2:
            # return dis1 * (e1 - s1 / 2 - e2 / 2) + dis2 * (e2 / 2 + s1 / 2 - s2)
            return dis1 + avg_dis2 * (s1 - s2)
        elif s1 >= e2:
            return dis1 + dis2
        else:
            return -1.0

    def _merge_match_frame(
        self,
        ref_name: str,
        new_res_list: list[QueryResult],
    ) -> None:

        if self.print_flag == True:
            could_ppp = True
        else:
            could_ppp = False

        # if ref_name != "1441021264-no-edge.dna":
        #     self.print_flag = False

        new_res_dict: dict[int, list[QueryResult]] = dict()
        for res in new_res_list:
            if res.ref_start_index not in new_res_dict.keys():
                new_res_dict[res.ref_start_index] = list()
            new_res_dict[res.ref_start_index].append(res)

        src_query_res_list = self.query_result_dict[ref_name]
        src_quick_search_list = self.quick_search_result_dict[ref_name]
        src_quick_search_remove_list = list()

        src_query_res_list_len: int = len(src_query_res_list)
        new_res_list_len: int = len(new_res_list)
        min_new_ref_start_index = -1

        if self.print_flag:
            log.info(
                f"quick search list: {src_quick_search_list}, src_query_res_list: {len(src_query_res_list)}"
            )
            log.info("src_res_list " * 5)
            for r in src_quick_search_list:
                r = self.query_result_dict[ref_name][r]
                log.info(
                    f"src s id: {r.sample_start_index} - {r.sample_end_index}, src ref id: {r.ref_start_index} - {r.ref_end_index}, remove: {r.remove_flag}"
                )

            log.info("new_res_list " * 5)
            for r in new_res_list:
                log.info(
                    f"new s id: {r.sample_start_index} - {r.sample_end_index}, new ref id: {r.ref_start_index} - {r.ref_end_index}"
                )
            log.info("=== " * 5)

        for i in src_quick_search_list:
            src_query_res = src_query_res_list[i]
            cur_ref_id = src_query_res.ref_end_index

            cur_sample_id = src_query_res.sample_end_index
            target_sample_id = cur_sample_id + 1

            candidate_query_result = None
            target_ref_id = cur_ref_id + 0
            for j in range(1, 2):
                tmp_target_ref_id = target_ref_id + j
                # log.info(f"tmp_target_ref_id: {tmp_target_ref_id}")
                if (
                    tmp_target_ref_id in new_res_dict.keys()
                    and new_res_dict[tmp_target_ref_id][0].sample_start_index
                    == target_sample_id
                ):
                    candidate_query_result = new_res_dict[tmp_target_ref_id][0]
                    break

            if candidate_query_result is None:
                src_quick_search_remove_list.append(i)
                continue

            src_query_res.distance += candidate_query_result.distance
            # src_query_res.ref_end_index = target_ref_id
            src_query_res.ref_end_index = target_ref_id + j
            src_query_res.sample_end_index = candidate_query_result.sample_end_index
            candidate_query_result.remove_flag = True

            if min_new_ref_start_index == -1:
                min_new_ref_start_index = target_ref_id
            else:
                min_new_ref_start_index = min(min_new_ref_start_index, target_ref_id)

        left_res_list = [
            new_res_list[i]
            for i in range(0, new_res_list_len)
            if new_res_list[i].ref_start_index > min_new_ref_start_index
            and new_res_list[i].remove_flag == False
        ]

        # if self.print_flag:
        #     log.info(f"left_res_list: {len(left_res_list)}")

        left_res_list_len = len(left_res_list)
        src_query_res_list.extend(left_res_list)

        # if self.print_flag:
        #     log.info(
        #         f"src_quick_search_remove_list 111: {src_quick_search_remove_list}"
        #     )

        #     log.info(f"src_quick_search_list 111: {src_quick_search_list}")

        for i in src_quick_search_remove_list:
            src_quick_search_list.remove(i)

        new_quick_search_list = [
            i
            for i in range(
                src_query_res_list_len, src_query_res_list_len + left_res_list_len
            )
        ]
        src_quick_search_list.extend(new_quick_search_list)
        # if self.print_flag:
        #     log.info(f"src_quick_search_list 222: {src_quick_search_list}")

        if self.print_flag:
            log.info("frame result " * 5)
            for r in src_quick_search_list:
                r = self.query_result_dict[ref_name][r]
                log.info(
                    f"result s id: {r.sample_start_index} - {r.sample_end_index}, result ref id: {r.ref_start_index} - {r.ref_end_index}, remove: {r.remove_flag}"
                )

    def merge_match_frame(
        self,
        new_res_dict: dict[str, list[QueryResult]],
    ):
        for new_ref_name in new_res_dict.keys():
            new_res_list = new_res_dict[new_ref_name]
            new_res_list.sort(key=lambda res: res.ref_start_index)
            if new_ref_name not in self.query_result_dict.keys():
                self.query_result_dict[new_ref_name] = list()
                self.query_result_dict[new_ref_name].extend(new_res_list)
                self.quick_search_result_dict[new_ref_name] = [
                    i for i in range(len(self.query_result_dict[new_ref_name]))
                ]
            else:
                self._merge_match_frame(new_ref_name, new_res_list)

    def _merge_match_segments(
        self,
        ref_name: str,
    ):

        merge_query_res_list: list[QueryResult] = list()
        query_result_list = self.query_result_dict[ref_name]
        query_result_list_len = len(query_result_list)

        for i in range(0, query_result_list_len):
            if query_result_list[i].remove_flag == True:
                if self.print_flag:
                    log.info(f"skip s: {query_result_list[i]}")
                continue

            merge_query_res = query_result_list[i]
            sample_threadhold = int(
                (
                    merge_query_res.sample_end_index
                    - merge_query_res.sample_start_index
                    + 1
                )
                * 0.5
            )

            if sample_threadhold <= 0:
                sample_threadhold = 5

            for j in range(i + 1, query_result_list_len):
                if query_result_list[j].remove_flag == True:
                    continue

                next_query_res = query_result_list[j]

                sample_lower_boud = (
                    next_query_res.sample_start_index - sample_threadhold
                )
                sample_upper_boud = next_query_res.sample_end_index + sample_threadhold

                if sample_lower_boud < 0:
                    sample_lower_boud = 0

                if self.print_flag:
                    log.info(
                        f"111, src s: {merge_query_res.sample_start_index} - {merge_query_res.sample_end_index}, next s: {sample_lower_boud} - {sample_upper_boud}, next ss: {next_query_res.sample_start_index } - {next_query_res.sample_end_index}, sample_threadhold: {sample_threadhold}, could merge: {self._could_merge(merge_query_res.sample_start_index,merge_query_res.sample_end_index,sample_lower_boud,sample_upper_boud)}"
                    )

                if self._could_merge(
                    merge_query_res.sample_start_index,
                    merge_query_res.sample_end_index,
                    sample_lower_boud,
                    sample_upper_boud,
                ):
                    ref_threadhold = int(
                        (
                            merge_query_res.ref_end_index
                            - merge_query_res.ref_start_index
                        )
                        * 0.5
                    )
                    if ref_threadhold <= 0:
                        ref_threadhold = 5 * 10

                    ref_lower_bound = next_query_res.ref_start_index - ref_threadhold
                    ref_upper_bound = next_query_res.ref_end_index + ref_threadhold

                    if ref_lower_bound < 0:
                        ref_lower_bound = 0

                    if self.print_flag:
                        log.info(
                            f"222, src r: {merge_query_res.ref_start_index} - {merge_query_res.ref_end_index}, next r: {ref_lower_bound} - {ref_upper_bound}, next rr: {next_query_res.ref_start_index} - {next_query_res.ref_end_index} could merge: {self._could_merge(merge_query_res.ref_start_index,merge_query_res.ref_end_index,ref_lower_bound,ref_upper_bound)}"
                        )

                    if self._could_merge(
                        merge_query_res.ref_start_index,
                        merge_query_res.ref_end_index,
                        ref_lower_bound,
                        ref_upper_bound,
                    ):
                        new_s_s = min(
                            merge_query_res.sample_start_index,
                            next_query_res.sample_start_index,
                        )
                        new_s_e = max(
                            merge_query_res.sample_end_index,
                            next_query_res.sample_end_index,
                        )

                        new_r_s = min(
                            merge_query_res.ref_start_index,
                            next_query_res.ref_start_index,
                        )
                        new_r_e = max(
                            merge_query_res.ref_end_index,
                            next_query_res.ref_end_index,
                        )

                        if self.print_flag:
                            log.info(
                                f"new s: {new_s_s} - {new_s_e}, new r: {new_r_s} - {new_r_e}, new_r_e - new_r_s == (new_s_e - new_s_s) * 10: {new_r_e - new_r_s == (new_s_e - new_s_s) * 10}"
                            )

                        next_query_res.remove_flag = True
                        if self.print_flag:
                            log.info(
                                f"res is removed: s: {next_query_res.sample_start_index} - {next_query_res.sample_end_index}, r: {next_query_res.ref_start_index} - {next_query_res.ref_end_index}"
                            )
                        merge_query_res.distance = self._merge_distance(
                            merge_query_res.ref_start_index,
                            merge_query_res.ref_end_index,
                            next_query_res.ref_start_index,
                            next_query_res.ref_end_index,
                            merge_query_res.distance,
                            next_query_res.distance,
                        )
                        merge_query_res.sample_start_index = new_s_s
                        merge_query_res.sample_end_index = new_s_e
                        merge_query_res.ref_start_index = new_r_s
                        merge_query_res.ref_end_index = new_r_e
                        sample_threadhold = int((new_s_e - new_s_s) * 0.5)
                else:
                    if merge_query_res.sample_end_index > sample_upper_boud:
                        break

            merge_query_res_list.append(merge_query_res)

        self.query_result_dict[ref_name].clear()
        self.query_result_dict[ref_name].extend(merge_query_res_list)
        if self.print_flag:
            log.info(" Merge Results -- " * 5)
            for i in merge_query_res_list:
                log.info(
                    f"s: {i.sample_start_index} - {i.sample_end_index}, r: {i.ref_start_index} - {i.ref_end_index}"
                )
        # time_cost += get_current_unix_timestamp() - t1

    def merge_match_segments(self):
        for ref_name in self.query_result_dict.keys():
            self._merge_match_segments(ref_name)

    def filter_valid_match_segment(self):
        for ref_name in self.query_result_dict.keys():
            src_list = self.query_result_dict[ref_name]
            if len(src_list) == 0:
                continue
            # src_list.sort(key=lambda res: res.sample_start_index)
            tmp_list = list()
            for i in range(len(src_list)):
                cur_res = src_list[i]
                if (
                    cur_res.sample_end_index - cur_res.sample_start_index + 1
                    >= self.segment_len_limit
                ):
                    tmp_list.append(cur_res)

            self.query_result_dict[ref_name][:] = []
            self.query_result_dict[ref_name].extend(tmp_list)

    def just_dump(self, with_time: bool):
        for ref_name in self.query_result_dict.keys():

            log.info(f"ref: {ref_name} -----")
            for res in self.query_result_dict[ref_name]:
                if not with_time:
                    log.info(
                        f"s: {res.sample_start_index} - {res.sample_end_index}, ref: {res.ref_start_index} - {res.ref_end_index}, remove: {res.remove_flag}"
                    )
                else:
                    s_s_t = res.sample_start_index * 125
                    s_e_t = (res.sample_end_index + 1) * 125
                    r_s_t = res.ref_start_index * 125
                    r_e_t = (res.ref_end_index + 1) * 125

                    s_s_t_t = milliseconds_to_hhmmss(s_s_t)
                    s_e_t_t = milliseconds_to_hhmmss(s_e_t)
                    r_s_t_t = milliseconds_to_hhmmss(r_s_t)
                    r_e_t_t = milliseconds_to_hhmmss(r_e_t)

                    log.info(
                        f"s: {s_s_t_t} - {s_e_t_t}, ref: {r_s_t_t} - {r_e_t_t}, remove: {res.remove_flag}"
                    )

            log.info("------\n")

    #     self,
    #     total_cost: float,
    #     search_cost: float,
    #     sample_des: MediaDes,
    # ):
    #     num = 0
    #     sample_len = sample_des.frame_total_len
    #     sample_name = sample_des.name
    #     sample_duration = sample_len * sample_des.avg_frame_time

    #     for ref_name, ref_list in self.query_result_dict.items():
    #         if len(ref_list) == 0:
    #             continue

    #         # if ref_name == "1441021264.vdna":
    #         log.info(" DUMP -- DUMP -- DUMP -- " * 5)
    #         for i in ref_list:
    #             log.info(
    #                 f"s: {i.sample_start_index} - {i.sample_end_index}, r: {i.ref_start_index} - {i.ref_end_index}"
    #             )

    #         total_match_ref_len = 0.0
    #         total_match_percent = 0.0
    #         total_match_sample_len = 0.0
    #         total_avg_dis = 0.0
    #         total_match_duration = 0.0
    #         last_s_s = -1
    #         last_s_e = -1
    #         last_r_e = -1
    #         match_log = "==== MATCH LOG ==== " * 10 + "\n"
    #         ref_len = ref_list[0].media_des.frame_total_len

    #         for ref in ref_list:
    #             match_ref_len = ref.ref_end_index - ref.ref_start_index + 1
    #             match_sample_len = ref.sample_end_index - ref.sample_start_index + 1
    #             match_percent = -1.0
    #             avg_dis = ref.distance / match_ref_len

    #             if sample_len != 0:
    #                 match_percent = match_ref_len / sample_len * 100.0

    #             if (
    #                 last_s_s <= ref.sample_start_index
    #                 and last_s_e >= ref.sample_start_index
    #             ):
    #                 continue

    #             if (
    #                 last_s_e <= ref.sample_start_index
    #                 and last_r_e <= ref.ref_start_index
    #             ) or (
    #                 last_s_e != ref.sample_end_index
    #                 and (last_r_e - ref.ref_start_index + 1) >= 0.5 * (ref_len)
    #             ):
    #                 sample_start_time = (
    #                     ref.sample_start_index * sample_des.avg_frame_time
    #                 )
    #                 sample_end_time = (
    #                     ref.sample_end_index + 1
    #                 ) * sample_des.avg_frame_time
    #                 ref_start_time = (
    #                     ref.ref_start_index
    #                 ) * ref.media_des.avg_frame_time
    #                 ref_end_time = (ref.ref_end_index) * ref.media_des.avg_frame_time

    #                 total_match_ref_len += match_ref_len
    #                 total_match_sample_len += match_sample_len
    #                 total_match_percent += match_percent
    #                 total_avg_dis += avg_dis
    #                 total_match_duration += sample_end_time - sample_start_time
    #                 match_log += f"{sample_name} match with {ref_name}, s: {ref.sample_start_index} - {ref.sample_end_index} r: {ref.ref_start_index} - {ref.ref_end_index} --- sample: {sample_start_time} - {sample_end_time} - d: {sample_end_time - sample_start_time }, ref: {ref_start_time} - {ref_end_time} - d: {ref_end_time - ref_start_time}\n"
    #                 last_s_s = ref.sample_start_index
    #                 last_s_e = ref.sample_end_index
    #                 last_r_e = ref.ref_end_index
    #             num += 1

    #         if (
    #             total_match_duration > 0.1 * sample_duration
    #             or ref_name.split(".")[0] in sample_name
    #         ):
    #             log.info(
    #                 match_log
    #                 + f"{sample_name} match with {ref_name}, total match duration: {int(total_match_duration + 0.5)} s, sample_duration: {sample_duration}\n"
    #             )

    #             if sample_name == ref_name and total_match_duration != sample_duration:
    #                 log.info(f"attention it, match rate too small: {sample_name}")

    #     if num == 0:
    #         log.info(f"sample_name: {sample_name} have no match, attetion it!!!!!")


def handler_query_results(
    milvus_client: MilvusClient,
    media_info_collection_name: str,
    l2_dis_thresh: float,
    results: list[list[dict[Any, Any]]],
    segment_len_limit: int,
    sample_name: str,
):
    query_result_handler = QueryResultHandler(24)
    # query_result_handler.print_flag = True
    media_des_cache_dict: dict[str, MediaDes] = dict()

    tmp_query_result_dict: dict[str, list[QueryResult]] = dict()

    for i in range(len(results)):
        tmp_query_result_dict.clear()
        # if i >= 80:
        #     break

        # log.info(f"results[i]: {results[i]}")
        for j in range(len(results[i])):
            l2_dis = results[i][j]["distance"]
            if l2_dis > l2_dis_thresh:
                break
            entity = results[i][j]["entity"]

            ref_frame = entity["frame"]
            ref_uuid = entity["uuid"]

            media_des = None

            if ref_uuid not in media_des_cache_dict.keys():
                ref_info = milvus_client.query(
                    media_info_collection_name,
                    filter=f'uuid in ["{ref_uuid}",]',
                    output_fields=["filename", "frame_total_len", "uuid"],
                )
                ref_file_name = ref_info[0]["filename"]
                frame_total_len = ref_info[0]["frame_total_len"]
                media_des = MediaDes(
                    name=ref_file_name,
                    frame_total_len=frame_total_len,
                    avg_frame_time=125.0,
                )
                media_des_cache_dict[ref_uuid] = media_des
            else:
                media_des = media_des_cache_dict[ref_uuid]

            if media_des == None:
                raise Exception("not found media des....")

            res = QueryResult(
                ref_start_index=ref_frame,
                ref_end_index=ref_frame,
                distance=l2_dis,
                sample_start_index=i,
                sample_end_index=i,
                # media_des=media_des,
            )

            if media_des.name not in tmp_query_result_dict.keys():
                tmp_query_result_dict[media_des.name] = list()
            tmp_query_result_dict[media_des.name].append(res)

        query_result_handler.merge_match_frame(tmp_query_result_dict)

    query_result_handler.filter_valid_match_segment()
    query_result_handler.merge_match_segments()
    query_result_handler.filter_valid_match_segment()

    log.info(f"sample name: {sample_name} matched with:::: ")
    query_result_handler.just_dump(True)

    # sample_des = MediaDes(
    #     name="xxx",
    #     frame_total_len=len(results),
    #     avg_frame_time=125.0,
    # )
    # query_result_handler.dump(-1, -1, sample_des)
